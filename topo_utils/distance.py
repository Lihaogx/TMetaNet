from persim import bottleneck, wasserstein, heat
from scipy.stats import entropy as scipy_entropy

def compute_persistent_entropy(diagram):
    """
    Calculate the persistent entropy of a persistence diagram.

    Args:
        diagram (np.ndarray): Persistence diagram with shape (n, 2) representing birth and death times.

    Returns:
        persistent_entropy (float): The persistent entropy value.
    """
    if len(diagram) == 0:
        return 0.0
    # Calculate persistence of each feature
    persistences = diagram[:, 1] - diagram[:, 0]
    
    # Normalize persistences to form probability distribution
    probabilities = persistences / persistences.sum()
    
    # Calculate persistent entropy
    persistent_entropy = scipy_entropy(probabilities)
    
    return persistent_entropy

def compute_entropy_list(topo_diagram, i):
    entropy_list = []
    for k in topo_diagram:
        entropy_list.append(compute_persistent_entropy(k[i]))
    return entropy_list
######################################################

#################### Wasserstein Distance ######################
def compute_wasserstein_distance(diagram1, diagram2):
    """
    Calculate the Wasserstein distance between two persistence diagrams.

    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram

    Returns:
        wasserstein_dist (float): The Wasserstein distance
    """
    wasserstein_dist_0 = 0.0 if len(diagram1[0]) == 0 or len(diagram2[0]) == 0 else wasserstein(diagram1[0], diagram2[0])
    wasserstein_dist_1 = 0.0 if len(diagram1[1]) == 0 or len(diagram2[1]) == 0 else wasserstein(diagram1[1], diagram2[1])
    wasserstein_dist = (wasserstein_dist_0 + wasserstein_dist_1) / 2
    return wasserstein_dist
#########################################################

##################### Bottleneck Distance #######################
def compute_bottleneck_distance(diagram1, diagram2):
    """
    Calculate the bottleneck distance between two persistence diagrams.

    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram

    Returns:
        bottleneck_dist (float): The bottleneck distance
    """
    bottleneck_dist = bottleneck(diagram1[0], diagram2[0])
    return bottleneck_dist
####################################################

######################### Heat Distance #####################
def compute_heat_distance(diagram1, diagram2):
    """
    Calculate the heat kernel distance between two persistence diagrams.

    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram

    Returns:
        heat_dist (float): The heat kernel distance
    """
    heat_dist = heat(diagram1[0], diagram2[0])
    return heat_dist
#######################################################