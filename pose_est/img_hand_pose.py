import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Path to the input image
IMAGE_FILE = r"C:\Users\Bhanu2003\Downloads\WhatsApp Image 2023-11-22 at 11.21.21 AM.jpeg"

# Read the input image
image = cv2.imread(IMAGE_FILE)
if image is None:
    print(f"Error: Unable to read the image file '{IMAGE_FILE}'")
    exit()

# Flip the image horizontally for correct handedness output
image = cv2.flip(image, 1)

# Convert the BGR image to RGB before processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe Hands
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    
    # Process the image
    results = hands.process(image_rgb)

    # Print handedness and draw hand landmarks on the image
    print('Handedness:', results.multi_handedness)
    if results.multi_hand_landmarks:
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save the annotated image
        cv2.imwrite('annotated_image.jpg', annotated_image)
        print("Annotated image saved as 'annotated_image.jpg'")
    else:
        print('No hands detected in the image.')
