import mediapipe as mp
import cv2

# Print versions of mediapipe and OpenCV
# print(f"mediapipe version: {mp.__version__}")
# print(f"OpenCV version: {cv2.__version__}")

# Initialize mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh  # 468 face landmarks
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Function to get face landmarks from an image
def get_landmarks(image):
    # Initialize FaceMesh model
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5)
    
    # Make the image unwriteable to improve performance
    image.flags.writeable = False
    
    # Process the image to get face landmarks
    result = face_mesh.process(image)
    
    # Get landmarks from the result
    landmarks = result.multi_face_landmarks[0].landmark
   
    
    return result, landmarks

# Function to draw landmarks on an image
def draw_landmarks(image, result):
    # Make the image writeable to draw on it
    image.flags.writeable = True
    
    # Check if there are faces in the result
    if result.multi_face_landmarks:
        # Draw landmarks on the image
        for face_landmark in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_list=face_landmark,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()


            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()

            )
    
    return image

# Path to the input image
path_img = 'data/photo3.jpg'
img = cv2.imread(path_img)

# Make a copy of the image for annotations
annotated_img = img.copy()

#   video code starts here
video_path ='data/giphy.gif'
cap= cv2.VideoCapture(video_path)
#  webcam settings
cap.set(10,100)
while True:
    success,frame = cap.read()
    if not success:
        print("Ignoring empty camera frame ")
        break  
    # for webcam use continue
    frame_cp = frame.copy()
    results, landmarks = get_landmarks(image=frame_cp)
    annotated_frame=draw_landmarks(frame_cp,results)
    cv2.imshow("Original video", frame)
    cv2.imshow("Annotated video", annotated_frame)

    if cv2.waitKey(5) & 0xFF== ord('q'):
        break


cap.release()   


# video code end here 

# Get face landmarks and results
result, landmarks = get_landmarks(image=img)

# Draw landmarks on the annotated image
annotated_img = draw_landmarks(image=annotated_img, result=result)

# Show original and annotated images
cv2.imshow("Original image", img)
cv2.imshow("Annotated image", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



