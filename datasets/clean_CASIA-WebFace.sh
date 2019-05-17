while read FACE_ID;
do
    echo "Removing CASIA-WebFace/$FACE_ID"
    rm -rf CASIA-WebFace/$FACE_ID
done < webface_lfw_overlap.txt
