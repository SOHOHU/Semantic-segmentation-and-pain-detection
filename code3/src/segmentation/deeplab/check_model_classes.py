
import torch
import os

def check_model():
    model_path = "model_data/deeplab_xception.pth"
    if not os.path.exists(model_path):
        print("Model not found")
        return

    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print("Model loaded.")
        
        keys = list(state_dict.keys())
        print(f"Total keys: {len(keys)}")
        for key in keys[:20]:
            print(f"{key}: {state_dict[key].shape}")
            
        # Search for specific last layer candidates
        candidates = [k for k in keys if 'classifier' in k or 'score' in k or 'last' in k]
        print("\nPossible classifier layers:")
        for k in candidates:
            print(f"{k}: {state_dict[k].shape}")

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model()
