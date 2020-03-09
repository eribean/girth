import numpy as np
from scipy.optimize import fminbound, fmin_powell

from girth import trim_response_set_and_counts, rasch_approx
from girth.synthetic import _graded_func


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


def graded_jml(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination paramaters
        for a graded response two parameter model

        difficulty parameters are category boundaries and by
        definition must be ordered

        Args:
            dataset: [items x participants] matrix of ordinal values
            max_iter: maximum number of iterations to run

        Returns:
            array of discriminations, array of difficulty estimates
    """
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    n_items, _ = unique_sets.shape

    # Remove the zero and full count values
    mask = np.var(unique_sets, axis=0) > 0
    unique_sets = unique_sets[:, mask]
    counts = counts[mask]

    # Set initial parameter estimates to default
    discrimination = np.ones((n_items,))
    min_value, max_value = unique_sets.max() - unique_sets.min()
    ordinal_range = max_value - min_value
    betas = np.ones((n_items, ordinal_range)) 
    betas *= np.linspace(-1, 1, ordinal_range)

    # Create a set of locations where response have occured
    n_takers = unique_sets.shape[1]
    thetas = np.zeros((n_takers,))
    response_locations = unique_sets - min_value

    # It may be the case that some rows don't have full responses
    # just remove responses and assume only n valid choices
    response_length = [np.unique(row).size for row in unique_sets]

    output_theta = np.zeros((ordinal_range, 1))
    output_parameters = np.zeros((ordinal_range, n_takers))


    for iteration in range(max_iter):
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas / alpha
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            def _theta_min(theta):
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
