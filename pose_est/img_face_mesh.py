import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILE = r"C:\Users\Bhanu2003\Downloads\WhatsApp Image 2023-11-22 at 11.48.04 AM.jpeg"
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:

    image = cv2.imread(IMAGE_FILE)
    if image is None:
        print(f"Error: Unable to read the image file '{IMAGE_FILE}'")
        exit()

    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
        print('No face detected in the image.')
    else:
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        output_path = 'annotated_image_static_1.png'
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved at: {output_path}")
