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