"""
🌿 FINAL PREDICTION SCRIPT - for 97% Model
Test your trained model on leaf images with top-3 predictions
and detailed prevention/treatment info.
"""

import tensorflow as tf
from tensorflow import keras # type: ignore
import numpy as np
from PIL import Image
import os

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

MODEL_PATH = 'final_best.keras'   # Your final model name
CLASS_NAMES_PATH = 'class_names.npy'
IMG_SIZE = (224, 224)

# ============================================================================ #
# DISEASE INFORMATION DATABASE
# ============================================================================ #
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'name': 'Apple Scab',
        'prevention': [
            'Prune trees to improve air circulation',
            'Remove and dispose of fallen leaves in autumn',
            'Avoid overhead watering',
            'Apply preventive fungicides during early spring',
            'Choose resistant apple varieties'
        ],
        'treatment': [
            'Use copper-based or sulfur fungicides early in the season',
            'Remove infected leaves and twigs immediately'
        ]
    },
    'Apple___Black_rot': {
        'name': 'Apple Black Rot',
        'prevention': [
            'Prune dead or infected branches',
            'Remove mummified fruits and leaves',
            'Maintain orchard cleanliness'
        ],
        'treatment': [
            'Use captan fungicide weekly',
            'Destroy infected fruits immediately'
        ]
    },
    'Tomato___Bacterial_spot': {
        'name': 'Tomato Bacterial Spot',
        'prevention': [
            'Use certified disease-free seeds or seedlings',
            'Avoid overhead irrigation to reduce leaf wetness',
            'Rotate crops for 2 years',
            'Disinfect tools regularly'
        ],
        'treatment': [
            'Apply copper-based bactericides weekly',
            'Remove infected leaves and fruits promptly',
            'Improve air circulation by pruning lower leaves'
        ]
    },
    'Tomato___Yellow_Leaf_Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus',
        'prevention': [
            'Control whiteflies with yellow sticky traps',
            'Use resistant tomato varieties',
            'Remove infected plants immediately'
        ],
        'treatment': [
            'No cure — focus on vector control',
            'Use neem oil for whiteflies'
        ]
    },
    'Tomato___mosaic_virus': {
        'name': 'Tomato Mosaic Virus',
        'prevention': [
            'Avoid touching healthy plants after infected ones',
            'Disinfect tools frequently',
            'Use certified virus-free seeds'
        ],
        'treatment': [
            'No cure — remove infected plants immediately',
            'Disinfect greenhouse and seed trays'
        ]
    },
    'Tomato___healthy': {
        'name': 'Healthy Tomato Plant',
        'prevention': [
            'Maintain good watering schedule',
            'Provide balanced nutrients',
            'Inspect plants weekly for early disease signs'
        ],
        'treatment': [
            'No treatment needed — keep up healthy habits!',
            'Monitor for any pest or discoloration early'
        ]
    }
}

# ============================================================================ #
# LOAD MODEL + CLASSES
# ============================================================================ #

print("=" * 70)
print("🌱 CROP DISEASE PREDICTOR (97% MODEL)")
print("=" * 70)

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

try:
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    print(f"✅ Found {len(class_names)} class labels.")
except:
    print("⚠️ class_names.npy not found — using generic labels.")
    class_names = [f"Class_{i}" for i in range(model.output_shape[-1])]

# ============================================================================ #
# PREDICTION FUNCTION
# ============================================================================ #

def predict_disease(image_path, top_k=3):
    """Predict disease from a single leaf image"""
    print(f"\n{'='*70}")
    print(f"🔍 Analyzing: {os.path.basename(image_path)}")
    print(f"{'='*70}")

    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Run prediction
    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    print("\n📊 TOP 3 PREDICTIONS:\n")

    for rank, idx in enumerate(top_indices, 1):
        disease_key = class_names[idx]
        confidence = predictions[idx] * 100

        info = DISEASE_INFO.get(disease_key, {
            'name': disease_key.replace('___', ' - ').replace('_', ' '),
            'prevention': ['Information not available'],
            'treatment': ['Consult agricultural expert']
        })

        tag = "🏆" if rank == 1 else f"#{rank}"
        print(f"{tag} {info['name']}")
        print(f"   Confidence: {confidence:.2f}%")

        if rank == 1:
            print(f"\n   📋 Description:")
            print(f"      Disease: {info['name']}\n")

            print(f"   🛡️  Prevention:")
            for i, p in enumerate(info['prevention'], 1):
                print(f"      {i}. {p}")

            print(f"\n   💊 Treatment:")
            for i, t in enumerate(info['treatment'], 1):
                print(f"      {i}. {t}")

        print()

# ============================================================================ #
# FOLDER PREDICTION
# ============================================================================ #

def predict_folder(folder_path, top_k=3):
    valid_ext = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in os.listdir(folder_path) if f.endswith(valid_ext)]

    if not image_files:
        print("❌ No images found in folder!")
        return

    print(f"\n✅ Found {len(image_files)} images in '{folder_path}'\n")
    for i, file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}]")
        predict_disease(os.path.join(folder_path, file), top_k=top_k)
        if i < len(image_files):
            input("Press Enter for next image...")

# ============================================================================ #
# INTERACTIVE MODE
# ============================================================================ #

def interactive_mode():
    while True:
        print("\n" + "="*70)
        print("🌿 Interactive Prediction Mode")
        print("="*70)
        print("1. Predict a single image")
        print("2. Predict all images in a folder")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            img_path = input("Enter image path: ").strip().strip('"\'')
            if os.path.exists(img_path):
                predict_disease(img_path)
            else:
                print(f"❌ File not found: {img_path}")

        elif choice == '2':
            folder_path = input("Enter folder path: ").strip().strip('"\'')
            if os.path.exists(folder_path):
                predict_folder(folder_path)
            else:
                print(f"❌ Folder not found: {folder_path}")

        elif choice == '3':
            print("\n👋 Goodbye, leaf doctor!")
            break
        else:
            print("❌ Invalid choice!")

# ============================================================================ #
# MAIN
# ============================================================================ #

if __name__ == "__main__":
    test_folder = "test_images"
    if os.path.exists(test_folder) and os.listdir(test_folder):
        print(f"\n✅ Found test images in '{test_folder}/'")
        print("1. Test all images in folder")
        print("2. Interactive mode")

        choice = input("\nEnter choice (1-2): ").strip()
        if choice == '1':
            predict_folder(test_folder)
        else:
            interactive_mode()
    else:
        print("\n💡 Tip: Add sample leaves in 'test_images/' to auto-run tests.")
        interactive_mode()
