import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a YOLO model with Ultralytics")
	parser.add_argument("--data", type=str, default=os.path.join("Tumor-Detection-8", "data.yaml"), help="Path to data.yaml")
	parser.add_argument("--model", type=str, default="yolo11n.pt", help="Model checkpoint or model name (e.g., yolo11n.pt, yolov9s.pt)")
	parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
	parser.add_argument("--imgsz", type=int, default=640, help="Image size")
	parser.add_argument("--batch", type=int, default=16, help="Batch size")
	parser.add_argument("--device", type=str, default="", help="Device string, e.g. '0' for GPU 0, 'cpu' for CPU")
	parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (Windows often needs 0 or 2)")
	parser.add_argument("--project", type=str, default=os.path.join("runs", "detect"), help="Project directory for runs")
	parser.add_argument("--name", type=str, default="train", help="Run name")
	parser.add_argument("--resume", action="store_true", help="Resume last training run in this project/name")
	parser.add_argument("--seed", type=int, default=0, help="Random seed")
	parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
	parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
	parser.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic augmentation N epochs before end")
	return parser.parse_args()


def main() -> None:
	try:
		from ultralytics import YOLO
	except Exception as exc:
		print(f"[ERROR] Ultralytics not available: {exc}", file=sys.stderr)
		sys.exit(1)

	args = parse_args()

	# Prepare model
	if args.resume:
		# Expect weights at project/name/weights/last.pt
		last_ckpt = os.path.join(args.project, args.name, "weights", "last.pt")
		if not os.path.isfile(last_ckpt):
			print(f"[ERROR] --resume specified but checkpoint not found: {last_ckpt}", file=sys.stderr)
			sys.exit(2)
		model = YOLO(last_ckpt)
	else:
		model = YOLO(args.model)

	# Train
	results = model.train(
		data=args.data,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device if args.device else None,
		project=args.project,
		name=args.name,
		workers=args.workers,
		seed=args.seed,
		patience=args.patience,
		lr0=args.lr0,
		close_mosaic=args.close_mosaic,
		verbose=True,
	)

	# Print key results path for convenience
	print("\n[OK] Training complete.")
	print(f"Results dir: {results.save_dir if hasattr(results, 'save_dir') else os.path.join(args.project, args.name)}")


if __name__ == "__main__":
	main()


