import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
from data.ps import PSCountingDataset
from data.gf import GFCountingDataset


def to_uint8(im, mean, std):
    img = (im - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
    img = np.clip(img, -2, 2)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img


def convert_dataset(pt_dir, out_dir, split, mean, std,
                    box_w=3, box_h=3):
    """
    Converts dataset with:
        data["im"]     -> (C,H,W)
        data["points"] -> (N,2)  (x,y pixel coordinates)

    into YOLO detection format with fixed-size boxes.
    """

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

    os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    pt_fps = sorted(os.listdir(pt_dir))

    for ptfp in tqdm(pt_fps):

        data = torch.load(os.path.join(pt_dir, ptfp))

        im = data["im"].numpy()              # (C,H,W)
        points = data["points"].numpy()     # (N,2)

        im_uint8 = to_uint8(im, mean, std)  # (C,H,W) uint8

        # Convert to HWC for saving
        im_uint8 = im_uint8.transpose(1, 2, 0)

        H, W, _ = im_uint8.shape

        yolo_labels = []

        for (x, y) in points:

            # Ensure float
            x = float(x)
            y = float(y)

            # Define box corners
            x1 = x - box_w / 2
            y1 = y - box_h / 2
            x2 = x + box_w / 2
            y2 = y + box_h / 2

            # Clip to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)

            bw = x2 - x1
            bh = y2 - y1

            # Skip invalid / collapsed boxes
            if bw <= 1 or bh <= 1:
                continue

            # Convert to YOLO format (normalized)
            xc = (x1 + x2) / 2 / W
            yc = (y1 + y2) / 2 / H
            bw = bw / W
            bh = bh / H

            yolo_labels.append(f"0 {xc} {yc} {bw} {bh}")

        # ---- File name ----
        fname = ptfp.replace(".pt", ".png")

        # ---- Save image ----
        cv2.imwrite(
            os.path.join(out_dir, "images", split, fname),
            im_uint8
        )

        # ---- Save label file ----
        label_path = os.path.join(
            out_dir,
            "labels",
            split,
            fname.replace(".png", ".txt")
        )

        with open(label_path, "w") as f:
            for line in yolo_labels:
                f.write(line + "\n")


def train(dataset_fp, name):
    # Load a pretrained model
    model = YOLO("yolo26n.pt")

    results = model.train(data=dataset_fp, epochs=200, imgsz=64, device="cuda:0", cos_lr=True, cache=True, name=name)


def evaluate(mean, std, ckpt_path):
    from train import evaluate
    model = YOLO(ckpt_path)
    model.to("cuda")
    H, W = 64, 64

    if "gf" in ckpt_path:
        test_dataset = GFCountingDataset(imsize=64, split="test")
    elif "ps" in ckpt_path:
        test_dataset = PSCountingDataset(imsize=64, split="test")
    else:
        raise ValueError

    preds = []
    targets = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            inp, target = test_dataset[i]
            inp = to_uint8(inp[:-1].numpy(), mean, std).astype(float) / 255
            inp = torch.from_numpy(inp).to("cuda")  # (3,H,W)

            results = model.predict(
                inp[:3][None,],
                conf=0.05,
                verbose=False
            )

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()

            binary = torch.zeros(H, W)
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                xc = int((x1 + x2) / 2)
                yc = int((y1 + y2) / 2)

                if 0 <= xc < W and 0 <= yc < H:
                    binary[yc, xc] = 1

            # # display
            # fig, axs = plt.subplots(1, 3)
            # mean = [73.2, 80.2, 72.7]
            # std = [39.1, 39.1, 41.4]
            # unnormed = inp[0, :3].cpu().numpy() * np.array(std).reshape(-1, 1, 1) + np.array(mean).reshape(-1, 1, 1)
            # unnormed = np.clip(unnormed, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            # axs[0].imshow(unnormed)
            # axs[1].imshow(tgt[0, 0].cpu().numpy(), cmap='hot')
            # axs[2].imshow(output[0, 0].cpu().numpy(), cmap='hot')
            # print(tgt[0, 0].sum().item(), output[0, 0].sum().item())
            # plt.show()

            preds.append(binary)
            targets.append(torch.Tensor(target))
    preds = torch.stack(preds)
    targets = torch.cat(targets)
    metrics = evaluate(preds, targets)
    print(metrics)
    return


def create_mixed_folder(root):
    import shutil
    for ext, folder in zip(["png", "txt"], ["images", "labels"]):
        dir_clean = os.path.join(root, folder, "train_clean")
        dir_noisy = os.path.join(root, folder, "unlabeled")
        dir_out = os.path.join(root, folder, "train_noisy")

        # Collect PNG files
        files_A = [f for f in os.listdir(dir_clean) if f.lower().endswith(f".{ext}")]
        files_B = [f for f in os.listdir(dir_noisy) if f.lower().endswith(f".{ext}")]

        counter = 0

        # ---- Copy A once ----
        for fname in tqdm(files_A, desc="Copying A"):
            src = os.path.join(dir_clean, fname)
            new_name = f"A_{counter:06d}.{ext}"
            dst = os.path.join(dir_out, new_name)
            shutil.copy2(src, dst)
            counter += 1

        # ---- Copy B nine times ----
        for dup_idx in range(9):
            for fname in tqdm(files_B, desc=f"Copying B (dup {dup_idx + 1}/9)"):
                src = os.path.join(dir_noisy, fname)
                new_name = f"B{dup_idx}_{counter:06d}.{ext}"
                dst = os.path.join(dir_out, new_name)
                shutil.copy2(src, dst)
                counter += 1

        print("\nDone.")


if __name__ == "__main__":
    # mean = [420, 600, 640, 2100]
    # std = [250, 340, 415, 1170]
    mean = [73.2, 80.2, 72.7, 105.7]
    std = [39.1, 39.1, 41.4, 53.5]
    split = "val"
    # convert_dataset("/scratch/gf2/pt/test/",
    #                 "/scratch/gf2/yolo/", split, mean, std)
    # train(dataset_fp="/scratch/gf2/yolo/clean.yaml", name="gf_clean")
    # create_mixed_folder("/scratch/maurice_counting/yolo/")
    evaluate(mean=mean, std=std, ckpt_path="runs/detect/gf_clean/weights/best.pt")
