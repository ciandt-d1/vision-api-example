# Vision API Example

This simple code example demonstrates how to use the Google Vision API to tag images.
1.  Extract the file test-images.tar.gz to project root folder.
2.  Setup environment
    ```bash
    virtualenv -p python3 venv
    source venv/bin/activate
    pip3 install -r requirement.txt
    ```
2.  Running example:
    ```bash
    export PROJECT=your-gcp-project
    python3 snippet.py ${PROJECT} images/dataset.csv --export_json True
    ```

3. Check results:
    This example saves image tags to Google Cloud Datastore and uploads the image to Google Cloud Storage. This can be useful to build an extension to allow searching images by tags.

    * Datastore:
      ![alt text](https://storage.googleapis.com/vision-api-example/staging/datastore.png)

    * GCS:
      ![alt text](https://storage.googleapis.com/vision-api-example/staging/gcs.png)

    * JSON files containing image tags will be exported to 'output' folder if the flag '--export_json' is True.