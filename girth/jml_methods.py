import numpy as np
from scipy.optimize import fminbound, fmin_powell, fmin_slsqp

from girth import trim_response_set_and_counts, rasch_approx
from girth import condition_polytomous_response, irt_evaluation


def rasch_jml(dataset, discrimination=1, max_iter=25):
    """
        Estimates difficulty parameters in an IRT model

        Args:
            dataset: [items x participants] matrix of True/False Values
            discrimination: scalar of discrimination used in model (default to 1)
            max_iter: maximum number of iterations to run

        Returns:
            array of difficulty estimates
    """
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    n_items, _ = unique_sets.shape

    # Use easy model to seed guess
    betas = rasch_approx(dataset, discrimination)

    # Remove the zero and full count values
    unique_sets, counts = trim_response_set_and_counts(unique_sets, counts)

    n_takers = unique_sets.shape[1]
    the_sign = discrimination * (-1)**unique_sets
    thetas = np.zeros((n_takers,))

    for iteration in range(max_iter):
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            def _theta_min(theta):
                otpt = 1.0  / (1.0 + np.exp(np.outer(the_sign[:, ndx], (theta - betas))))

                return -np.log(otpt).sum()

            # Solves for the ability for each person
            thetas[ndx] = fminbound(_theta_min, -6, 6)

        # Recenter theta to identify model
        thetas -= thetas.mean()
        thetas /= thetas.std(ddof=1)

        #####################
        # STEP 2
        # Estimate Betas, given Theta
        # Loops over all items
        #####################
        for ndx in range(n_items):
            def _beta_min(beta):
                otpt = 1.0 / (1.0 + np.exp((thetas - beta) * the_sign[ndx,:]))
                return -np.log(otpt).dot(counts)

            # Solves for the beta parameters
            betas[ndx] = fminbound(_beta_min, -6, 6)

        if(np.abs(previous_betas - betas).max() < 1e-3):
            break

    return betas


def onepl_jml(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination paramters in 1PL IRT model

        Args:
            dataset: [items x participants] matrix of True/False Values
            max_iter: maximum number of iterations to run

        Returns:
            discrimination, array of difficulty estimates
    """
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    n_items, _ = unique_sets.shape

    # Use easy model to seed guess
    discrimination = 1.0
    betas = rasch_approx(dataset, discrimination)

    # Remove the zero and full count values
    unique_sets, counts = trim_response_set_and_counts(unique_sets, counts)

    n_takers = unique_sets.shape[1]
    the_sign = (-1)**unique_sets
    thetas = np.zeros((n_takers,))

    for iteration in range(max_iter):
        previous_discrimination = discrimination * 1.0

        #####################
        # STEP 1
        # Estimate theta, given betas / alpha
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            def _theta_min(theta):
                otpt = 1.0  / (1.0 + np.exp(np.outer(the_sign[:, ndx] * discrimination,
                                                     (theta - betas))))

                return -np.log(otpt).sum()
            # Solve for ability with each paramter
            thetas[ndx] = fminbound(_theta_min, -6, 6)

        # Recenter theta to identify model
        thetas -= thetas.mean()
        thetas /= thetas.std(ddof=1)

        #####################
        # STEP 2
        # Estimate Betas / alpha, given Theta
        # Loops over all items
        #####################
        def _alpha_min(estimate):
            # Initialize cost evaluation to zero
            cost = 0
            for ndx in range(n_items):
                def _beta_min(beta):
                    otpt = 1.0 / (1.0 + np.exp((thetas - beta) * the_sign[ndx,:] * estimate))
                    return -np.log(otpt).dot(counts)

                # Solves for the difficulty parameter for a given item at
                # a specific discrimination parameter
                betas[ndx] = fminbound(_beta_min, -6, 6)
                cost += _beta_min(betas[ndx])
            return cost

        # Solves for the single discrimination term
        discrimination = fminbound(_alpha_min, 0.25, 5)

        # Check termination conditions
        if(np.abs(previous_discrimination - discrimination).max() < 1e-3):
            break

    return discrimination, betas


def twopl_jml(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination paramters in 2PL IRT model

        Args:
            dataset: [items x participants] matrix of True/False Values
            max_iter: maximum number of iterations to run

        Returns:
            array of discriminations, array of difficulty estimates
    """
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    n_items, _ = unique_sets.shape

    # Use easy model to seed guess
    discrimination = np.ones((n_items,))
    betas = rasch_approx(dataset, discrimination)

    # Remove the zero and full count values
    unique_sets, counts = trim_response_set_and_counts(unique_sets, counts)

    n_takers = unique_sets.shape[1]
    the_sign = (-1)**unique_sets
    thetas = np.zeros((n_takers,))

    for iteration in range(max_iter):
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas / alpha
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            def _theta_min(theta):
                otpt = 1.0  / (1.0 + np.exp(np.outer(the_sign[:, ndx],
                                                     discrimination * (theta - betas))))

                return -np.log(otpt).sum()
            # Solves for ability parameters
            thetas[ndx] = fminbound(_theta_min, -6, 6)

        # Recenter theta to identify model
        thetas -= thetas.mean()
        thetas /= thetas.std(ddof=1)

        #####################
        # STEP 2
        # Estimate Betas / alpha, given Theta
        # Loops over all items
        #####################
        for ndx in range(n_items):
            def _alpha_beta_min(estimates):
                otpt = 1.0 / (1.0 + np.exp((thetas - estimates[1]) *
                                            the_sign[ndx,:] * estimates[0]))
                return -np.log(otpt).dot(counts)

            # Solves jointly for parameters using derivative free methods
            otpt = fmin_powell(_alpha_beta_min, (discrimination[ndx], betas[ndx]),
                               disp=False)
            discrimination[ndx], betas[ndx] = otpt

        # Check termination criterion
        if(np.abs(previous_betas - betas).max() < 1e-3):
            break

    return discrimination, betas


def _jml_inequality(test):
    """Inequality constraints for graded jml minimization."""
    # First position is discrimination, next are difficulties
    return np.concatenate(([1, 1], np.diff(test)[1:]))


def grm_jml(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination paramaters
        for a graded response two parameter model

        difficulty parameters are category boundaries and by
        definition must be ordered

        Args:
            dataset: [items x participants] matrix of ordinal values
            max_iter: maximum number of iterations to run

        Returns:
            array of discriminations, 
            array of difficulty estimates (np.nan is a null value)
            array of person ability estimates
    """
    responses, item_counts = condition_polytomous_response(dataset)    
    n_items, n_takers = responses.shape

    # Set initial parameter estimates to default
    thetas = np.zeros((n_takers,))

    # Initialize difficulty parameters for iterations
    betas = np.full((item_counts.sum(),), -10000.0)
    discrimination = np.ones_like(betas)
    cumulative_item_counts = item_counts.cumsum()
    start_indices = np.roll(cumulative_item_counts, 1)
    start_indices[0] = 0

    for ndx in range(n_items):
        end_ndx = cumulative_item_counts[ndx]
        start_ndx = start_indices[ndx] + 1
        betas[start_ndx:end_ndx] = np.linspace(-1, 1, 
                                               item_counts[ndx] - 1)
    betas_roll = np.roll(betas, -1)
    betas_roll[cumulative_item_counts-1] = 10000
    
    for iteration in range(max_iter):
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas / alpha
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            def _theta_min(theta):
                # Solves for ability parameters (theta)
                graded_prob = (irt_evaluation(betas, discrimination, theta) - 
                               irt_evaluation(betas_roll, discrimination, theta))

                values = graded_prob[responses[:, ndx]]
                return -np.log(values).sum()

            thetas[ndx] = fminbound(_theta_min, -6, 6)

        # Recenter theta to identify model
        thetas -= thetas.mean()
        thetas /= thetas.std(ddof=1)
        #####################
        # STEP 2
        # Estimate Betas / alpha, given Theta
        # Loops over all items
        #####################
        for ndx in range(n_items):
            # Compute ML for static items
            start_ndx = start_indices[ndx]
            end_ndx = cumulative_item_counts[ndx]
            graded_prob = (irt_evaluation(betas, discrimination, thetas) - 
                           irt_evaluation(betas_roll, discrimination, thetas))

            static_component = np.take_along_axis(graded_prob, 
                                                  np.delete(responses, ndx, axis=0), 
                                                  axis=0)
            partial_maximum_likelihood = -np.log(static_component).sum()
            
            def _alpha_beta_min(estimates):
                # Set the estimates int
                discrimination[start_ndx:end_ndx] = estimates[0]
                betas[start_ndx+1:end_ndx] = estimates[1:]
                betas_roll[start_ndx:end_ndx-1] = estimates[1:]

                graded_prob = (irt_evaluation(betas, discrimination, thetas) - 
                               irt_evaluation(betas_roll, discrimination, thetas))
                
                values = np.take_along_axis(graded_prob, responses[None, ndx], axis=0)
                np.clip(values, 1e-23, np.inf, out=values)
                return -np.log(values).sum() + partial_maximum_likelihood

            # Solves jointly for parameters using derivative free methods
            initial_guess = np.concatenate(([discrimination[start_ndx]], 
                                            betas[start_ndx+1:end_ndx]))
            otpt = fmin_slsqp(_alpha_beta_min, initial_guess,
                              disp=False, f_ieqcons=_jml_inequality,
                              bounds=[(.25, 4)] + [(-6, 6)] * (item_counts[ndx]-1))
            
            discrimination[start_ndx:end_ndx] = otpt[0]
            betas[start_ndx+1:end_ndx] = otpt[1:]
            betas_roll[start_ndx:end_ndx-1] = otpt[1:]
            
        # Check termination criterion
        if(np.abs(previous_betas - betas).max() < 1e-3):
            break

    # Trim difficulties to conform to standard output
    #TODO:  look where missing values are and place NAN there instead
    # of appending them to the end
    output_betas = np.full((n_items, item_counts.max()-1), np.nan)
    for ndx, (start_ndx, end_ndx) in enumerate(zip(start_indices, cumulative_item_counts)):
        output_betas[ndx, :end_ndx-start_ndx-1] = betas[start_ndx+1:end_ndx]
        
    
    return discrimination[start_indices], output_betas, thetas