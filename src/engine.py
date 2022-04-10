import torch

from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    train_dataset,
    batch_size,
    epochs,
    model,
    criterion,
    optimizer,
    val_dataset=None,
    scheduler=None,
    shuffle=True,
    device="cpu",
    logger=None,
):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    optimizer.zero_grad()
    total_len = len(train_dataloader)
    if val_dataset:
        total_len += len(val_dataloader)

    for epoch in range(1, epochs + 1):
        progress_bar = tqdm(total=total_len, desc=f"Epoch {epoch}/{epochs}")

        # Train
        model.train()
        train_loss = []
        for i, data in enumerate(train_dataloader):
            # Predict -> Calc Loss -> Step Optimizer
            labels = data["labels"].to(device)
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(torch.swapaxes(outputs.logits, 1, 2), labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            train_loss.append(loss.item())
            progress_bar.set_postfix({"Train Loss": sum(train_loss) / len(train_loss)})
            progress_bar.update()

            logger.log(
                data={"train loss": sum(train_loss) / len(train_loss)},
                step=(epoch - 1) * len(train_dataloader) + (i + 1),
            )

        # Val
        if val_dataset:
            model.eval()  # Set's dropout and other training specific variables to 0
            val_loss = []
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):
                    labels = data["labels"].to(device)
                    input_ids = data["input_ids"].to(device)
                    attention_mask = data["attention_mask"].to(device)
                    token_type_ids = data["token_type_ids"].to(device)

                    outputs = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(torch.swapaxes(outputs.logits, 1, 2), labels)

                    val_loss.append(loss.item())

                    progress_bar.set_postfix(
                        {
                            "Train Loss": sum(train_loss) / len(train_loss),
                            "Val Loss": sum(val_loss) / len(val_loss),
                        }
                    )
                    progress_bar.update()
                    logger.log(
                        data={"val loss": sum(val_loss) / len(val_loss)},
                        step=(epoch - 1) * len(val_dataloader) + (i + 1),
                    )

            progress_bar.close()

    return model, optimizer
