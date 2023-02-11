import os
import torch
from tqdm.auto import tqdm


def create_submission(experiment_name, model, loader, step_days=10):

    batch_size = loader.batch_size
    num_days_test = loader.dataset.ntimes

    chunk_size = batch_size * num_days_test

    if not os.path.exists("submissions/"):
        os.makedirs("submissions/")

    csv_submission = open(f"submissions/{experiment_name}.csv", "w")
    csv_submission.write("Id,Predicted\n")

    t_offset = 0
    submission_offset = 0

    for batch in tqdm(loader):
        positions, features = batch

        positions = positions.to(model.device)
        features = features.to(model.device)

        with torch.no_grad():
            predictions = model(positions, features)

        predictions = predictions.view(-1)

        yearcut_indices = list(range(0, chunk_size + t_offset, num_days_test))

        subdays_indices = [
            y + k
            for y in yearcut_indices
            for k in range(0, num_days_test, step_days)
        ]

        subdays_indices = list(map(lambda i: i - t_offset, subdays_indices))

        subdays_indices = [
            k
            for k in subdays_indices
            if 0 <= k < min(chunk_size, predictions.shape[0])
        ]

        t_offset = chunk_size - (yearcut_indices[-1] - t_offset)

        predictions_list = predictions[subdays_indices].tolist()

        submission_part = "\n".join(
            [
                f"{i+submission_offset},{pred}"
                for i, pred in enumerate(predictions_list)
            ]
        )

        csv_submission.write(submission_part + "\n")

        submission_offset += len(predictions_list)

    csv_submission.close()
