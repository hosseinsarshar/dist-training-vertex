curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @sd-pix2pix-tutorial-2node.json \
     "https://us-east4-aiplatform.googleapis.com/v1/projects/google.com:vertex-training-dlexamples/locations/us-east4/customJobs"
