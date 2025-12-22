import torch
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import custom ViTBlock
from custom_module_radpose import RADPoseBlock

# --- CRITICAL FIX: Proper module injection ---
from ultralytics.nn.tasks import parse_model

# Method 1: Patch the module dictionary directly
import ultralytics.nn.modules as modules

modules.RADPoseBlock = RADPoseBlock

# Method 2: Override parse_model's globals
if hasattr(parse_model, '__globals__'):
    parse_model.__globals__['ViTBlock'] = RADPoseBlock
else:
    # For older Python versions
    parse_model.func_globals['ViTBlock'] = RADPoseBlock

# Now import YOLO after custom module registration
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Alternative registration method
DetectionModel.ViTBlock = RADPoseBlock

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Verify custom module registration
    print("Registered modules:", [k for k in dir(modules) if k.startswith('V')])

    try:
        # Load model with custom architecture
        model = YOLO('yolov8-radpose.yaml').load('yolov8s-pose.pt')
        #model = YOLO('yolov8b-pose-vit.yaml').load('Vit-best.pt')
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Trying fallback initialization...")
        # Fallback: Create model directly
        from ultralytics.nn.tasks import attempt_load_weights

        model = DetectionModel('yolov8-radpose.yaml')
        #attempt_load_weights(model, 'Vit-best.pt')

    # Verify custom module in model architecture
    if hasattr(model, 'model'):
        print("Verifying ViTBlock integration:")
        vit_found = False
        for name, layer in model.model.named_modules():
            if isinstance(layer, RADPoseBlock):
                print(f"  ✓ ViTBlock found at: {name}")
                vit_found = True
        if not vit_found:
            print("  ✗ ViTBlock not found in model architecture!")

    # Training configuration
    train_args = {
        'data': 'coco-pose.yaml',
        'epochs': 100,
        'patience': 80,
        'imgsz': 640,
        'batch': 16,
        'verbose' : True,
        'device': device.type,
        'optimizer': 'AdamW',
        'lr0': 0.0003,
        'weight_decay': 0.05,
    }

    # Start training
    results = model.train(**train_args)

    # Export model
    model.export(format='onnx')