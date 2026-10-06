import os
import cv2
import torch
import numpy as np

from model import UNet


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using Device:", device)


# ============================================================
# PATHS
# ============================================================

VIDEO_DIR = "/Users/abhyudaysingh/Downloads/videos 2"

MODEL_PATH = "/Users/abhyudaysingh/unet_model.pth"

OUTPUT_DIR = "/Users/abhyudaysingh/Downloads/video_predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading U-Net...")

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# GET VIDEOS
# ============================================================

videos = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(".avi")
])

print("Videos found:", len(videos))

for video in videos:
    print("  ", video)


# ============================================================
# PROCESS EACH VIDEO
# ============================================================

for video_name in videos:

    input_path = os.path.join(
        VIDEO_DIR,
        video_name
    )

    output_name = os.path.splitext(video_name)[0] + "_prediction.mp4"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    print()
    print("=" * 60)
    print("Processing:", video_name)
    print("=" * 60)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("ERROR: Could not open video")
        continue


    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 10

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("Original size:", width, "x", height)
    print("FPS:", fps)
    print("Total frames:", total_frames)


    # --------------------------------------------------------
    # OUTPUT VIDEO
    #
    # We create:
    #
    # ORIGINAL | PREDICTION | OVERLAY
    # --------------------------------------------------------

    output_width = 512 * 3
    output_height = 512

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height)
    )


    frame_number = 0


    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1


        # ----------------------------------------------------
        # Convert video frame to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Resize to model input size
        # ----------------------------------------------------

        gray_resized = cv2.resize(
            gray,
            (512, 512),
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # Normalize exactly like training
        # ----------------------------------------------------

        image = gray_resized.astype(
            np.float32
        )

        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:

            image = (
                image - image_min
            ) / (
                image_max - image_min
            )

        else:

            image = np.zeros_like(
                image
            )


        # ----------------------------------------------------
        # Convert to PyTorch tensor
        #
        # Shape:
        #
        # [1, 1, 512, 512]
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image
        ).unsqueeze(0).unsqueeze(0)

        image_tensor = image_tensor.to(
            device
        )


        # ====================================================
        # U-NET PREDICTION
        # ====================================================

        with torch.inference_mode():

            output = model(
                image_tensor
            )

            probability = torch.sigmoid(
                output
            )

            prediction = (
                probability > 0.5
            ).float()


        # ----------------------------------------------------
        # Convert prediction to NumPy
        # ----------------------------------------------------

        prediction = (
            prediction[0, 0]
            .cpu()
            .numpy()
        )

        probability_np = (
            probability[0, 0]
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Binary segmentation mask
        # ----------------------------------------------------

        mask = (
            prediction * 255
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        original_display = cv2.cvtColor(
            gray_resized,
            cv2.COLOR_GRAY2BGR
        )


        # ----------------------------------------------------
        # Prediction display
        # ----------------------------------------------------

        prediction_display = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )


        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = original_display.copy()

        # Red where prediction exists

        red_mask = (
            prediction > 0
        )

        overlay[red_mask] = (
            0,
            0,
            255
        )

        # Blend original + segmentation

        overlay = cv2.addWeighted(
            original_display,
            0.65,
            overlay,
            0.35,
            0
        )


        # ----------------------------------------------------
        # Add labels
        # ----------------------------------------------------

        cv2.putText(
            original_display,
            "Original",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            prediction_display,
            "Prediction",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            "Overlay",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Combine three views
        # ----------------------------------------------------

        combined = np.hstack([
            original_display,
            prediction_display,
            overlay
        ])


        # ----------------------------------------------------
        # Write frame
        # ----------------------------------------------------

        writer.write(
            combined
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_number % 50 == 0:

            print(
                f"Processed "
                f"{frame_number}/"
                f"{total_frames} frames"
            )


    # ========================================================
    # CLOSE VIDEO
    # ========================================================

    cap.release()
    writer.release()

    print()
    print("Finished:", video_name)
    print("Saved to:")
    print(output_path)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("ALL VIDEOS PROCESSED")
print("=" * 60)
print()
print("Output folder:")
print(OUTPUT_DIR)
print()