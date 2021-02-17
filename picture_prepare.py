import face_recognition
import pickle
import cv2
import os

# lists of known face encodings and names
known_face_encodings = []
known_face_names = []


# Save the faces encodings
def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        faces_data = [known_face_encodings, known_face_names]
        pickle.dump(faces_data, face_data_file)
        print("Known faces backed up to disk.")


# Load the pictures and find the faces encodings
def load_and_find_faces(file_pathname):
    for filename in os.listdir(file_pathname):
        # print(filename)
        known_face_names.append(filename[:-4])
        image = cv2.imread(file_pathname+'/'+filename)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        image = image[:, :, ::-1]
        # Find the encodings
        # face_location = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
        face_encoding = face_recognition.face_encodings(image)[0]
        # expend the known_face_encodings
        known_face_encodings.append(face_encoding)


def main():
    load_and_find_faces("faces_data")
    # print(known_face_names)
    # print(known_face_encodings)
    save_known_faces()


if __name__ == '__main__':
    main()




