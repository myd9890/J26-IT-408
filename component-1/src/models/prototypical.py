import torch
import torch.nn.functional as F


def create_prototypes(
    embeddings,
    labels,
    num_classes
):
    """
    Create one prototype per class.

    A prototype is the mean embedding
    of the support examples belonging
    to that class.
    """

    prototypes = []

    for class_id in range(num_classes):

        class_embeddings = embeddings[
            labels == class_id
        ]

        prototype = class_embeddings.mean(
            dim=0
        )

        prototypes.append(prototype)

    return torch.stack(prototypes)


def classify_with_prototypes(
    query_embeddings,
    prototypes
):
    """
    Classify query embeddings using
    Euclidean distance to each prototype.
    """

    distances = torch.cdist(
        query_embeddings,
        prototypes
    )

    predictions = distances.argmin(
        dim=1
    )

    return predictions, distances