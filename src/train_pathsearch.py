import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
from datetime import datetime
from rich import print as rprint
from tqdm import tqdm
import argparse
from PathSearch import config


from PathSearch.model.PathSearch import PathSearch
from PathSearch.dataset.PathSearchTrainingDataset import PathSearchDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train PathSearch Model")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=8e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for optimizer')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--sample_num', type=int, default=512, help='Number of samples')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--log_interval', type=int, default=20, help='Interval for logging training status')
    parser.add_argument('--save_dir', type=str, default=str(config.RESULTS_DIR / 'weights' / 'step2' / 'PathSearch_hierarchical_bs128'), help='Directory to save models')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use for training')
    parser.add_argument('--img_sup_loss_weight', type=float, default=0.1, help='Weight for img supervised contrastive loss')
    parser.add_argument('--text_sup_loss_weight', type=float, default=0.1, help='Weight for text supervised contrastive loss')
    return parser.parse_args()

def evaluate(model, dataloader, device):
    """"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_pairs = 0
    total_sup_image_loss = 0.0
    total_sup_text_loss = 0.0

    with torch.no_grad(), torch.cuda.amp.autocast():
        for _, images, texts, _, _, _ in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            texts = texts.squeeze(1).to(device)

            image_features, mosaics, text_features, logit_scale = model(image=images, text=texts)

            logits_per_image = logit_scale * torch.matmul(image_features, text_features.T)
            logits_per_text = logit_scale * torch.matmul(text_features, image_features.T)
            labels = torch.arange(images.size(0), device=device)
            loss_contrast = (F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)) / 2

            loss = loss_contrast + cosine_diversity_loss(mosaics)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            preds = logits_per_image.argmax(dim=1)
            correct_pairs += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    avg_sup_image_loss = total_sup_image_loss / total_samples
    avg_sup_text_loss = total_sup_text_loss / total_samples
    accuracy = correct_pairs / total_samples

    logging.info(
        "Validation Loss: %.4f | Accuracy: %.2f%% | Contrast Loss: %.4f | Sup Image Loss: %.4f | Sup Text Loss: %.4f",
        avg_loss, accuracy * 100.0, loss_contrast.item(), avg_sup_image_loss, avg_sup_text_loss
    )

    return avg_loss, accuracy

def cosine_diversity_loss(mosaic):
    """
    mosaic: (B, K, D)
    Returns scalar loss to penalize high similarity between different vectors in mosaic
    """
    mosaic_norm = F.normalize(mosaic, dim=-1)  # normalize along D
    sim_matrix = torch.matmul(mosaic_norm, mosaic_norm.transpose(1, 2))  # (B, K, K)
    eye = torch.eye(sim_matrix.size(-1), device=mosaic.device).unsqueeze(0)
    loss = (sim_matrix * (1 - eye)).mean()  # exclude diagonal
    return loss

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    args.save_dir = args.save_dir + f"training_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_sn{args.sample_num}_imgsup{args.img_sup_loss_weight}_textsup{args.text_sup_loss_weight}"
    
    # Initialize logging
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, 
        f"training_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_sn{args.sample_num}_epochs{args.epochs}_"
        + datetime.now().strftime("%Y%m%d%H%M%S") + ".log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Build training and validation datasets
    # Use env/config driven defaults for data and cache directories so the code
    # works across machines. Users may override with PATHSEARCH_DATA_DIR
    default_data_path = os.environ.get('PATHSEARCH_DATA_DIR', str(config.DATA_DIR))
    default_cache_dir = os.environ.get('PATHSEARCH_CACHE_DIR', str(config.CACHE_DIR / 'step2_new' / 'sample_num512'))

    train_dataset = PathSearchDataset(
        path=default_data_path,
        mode='train',
        sample_num=args.sample_num,
        cache_dir=default_cache_dir,
        rebuild_cache=False
    )

    val_dataset = PathSearchDataset(
        path=default_data_path,
        mode='test',
        sample_num=args.sample_num,
        cache_dir=default_cache_dir,
        rebuild_cache=False
    )
    rprint("Train dataset size:", len(train_dataset))
    rprint("Validation dataset size:", len(val_dataset))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=PathSearchDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=PathSearchDataset.collate_fn,
    )

    model = PathSearch(embed_dim=args.embed_dim).to(args.device)
    # print(model.text_model)
    
    # Freeze selected parameters
    for param in model.text_model.visual.parameters():
        param.requires_grad = False
    for param in model.text_model.parameters():
        param.requires_grad = False
    for layer_idx in range(9, 12):
        for param in model.text_model.text.transformer.encoder.layer[layer_idx].parameters():
            param.requires_grad = True
    for param in model.text_model.text.proj.parameters():
        param.requires_grad = True

    # Print trainable parameters
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    rprint("Trainable Parameters:")
    for p in trainable_params:
        rprint(p)
    
    model = torch.nn.DataParallel(model)
    model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (_, images, texts, _, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
            images = images.to(args.device, non_blocking=True)
            texts = texts.squeeze(1).to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                image_features, mosaics, text_features, logit_scale = model(image=images, text=texts)
                # -
                logits_per_image = logit_scale.mean() * torch.matmul(image_features, text_features.T)
                logits_per_text = logit_scale.mean() * torch.matmul(text_features, image_features.T)
                labels = torch.arange(images.size(0), device=args.device)
                loss_contrast = (F.cross_entropy(logits_per_image, labels) +
                                 F.cross_entropy(logits_per_text, labels)) / 2

                loss_diversity = cosine_diversity_loss(mosaics)
                
                loss=loss_contrast + loss_diversity

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

                # Log three losses for monitoring
            if batch_idx % args.log_interval == 0:
                avg_loss = train_loss / (batch_idx + 1)
                logging.info(
                    "Epoch[%d/%d] Batch[%d/%d] Total Loss: %.4f | Contrast Loss: %.4f | Diversity Loss: %.4f",
                    epoch, args.epochs, batch_idx, len(train_loader), avg_loss, loss_contrast.item(), loss_diversity.item()
                )

    # ======= Validation Phase =======
        best_model_path = os.path.join(args.save_dir, "best_model_"+str(epoch)+".pt")
        val_loss, val_acc = evaluate(model, val_loader, args.device)
        logging.info(
            "Validation Epoch[%d] Loss: %.4f | Accuracy: %.2f%%",
            epoch, val_loss, val_acc * 100.0
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
                # remove other best model files
            for file in os.listdir(args.save_dir):
                if file.startswith("best_model"):
                    os.remove(os.path.join(args.save_dir, file))
            torch.save(model.state_dict(), best_model_path)
            logging.info("New best model saved with val loss: %.4f", val_loss)

    # Save checkpoints if needed
        # checkpoint = {
        #     "epoch": epoch,
        #     "model": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        #     "val_loss": val_loss,
        #     "val_acc": val_acc
        # }
        # if epoch % 50 == 0:
        #     torch.save(
        #         checkpoint,
        #         os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pt")
        #     )

if __name__ == "__main__":
    main()
