import glob
from django.shortcuts import redirect, render
from .models import Document
from .forms import DocumentForm
import shutil
import sys
from pathlib import Path
import tensorflow as tf
import ctypes
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute().parent.absolute()))

from scripts_notebooks import predict_keras_mlp_model

#PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary.dll"

#MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)

model = tf.keras.models.load_model("../models/keras_models/model_d.h5")

CLASSES = ["espagne", "france", "japon"]
CLASSES_SIZE = len(CLASSES)


def img_not_found(request):

    return render(request, 'img_not_found.html')


def error(request):
    try:
        shutil.rmtree("media\\documents")
    except:
        pass

    return render(request, 'error.html')


def analyze(request):
    try:
        # Load documents for the list page
        documents = Document.objects.all()
        document = documents.last

        # Render list page with the documents and the form
        result = "Résultat de l'analyse"
        file = ""

        for files in glob.iglob('media\\documents\\**\\*.*', recursive=True):
            file = files

        print("\n FILE : ", file, "\n")


        folder = file.rsplit('\\', 1)[0]
        image_file = file.rsplit('\\', 1)[1]
        image_class = predict_keras_mlp_model.predict(model,CLASSES,CLASSES_SIZE,file)
        try:
            context = {'documents': documents, "file_name": file.split('\\')[-1], 'document': document,
                       'result': result, 'predict': image_class}
            try:
                shutil.rmtree("media\\documents")
            except:
                pass

            return render(request, 'result.html', context)

        except:
            return redirect('error')
    except:
        return redirect('img_not_found')


def my_view(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            try:
                shutil.rmtree("media\\documents")
            except:
                pass
            newdoc.save()

            # Redirect to the document list after POST
            return redirect('my-view')
        else:
            print("Formulaire non valide")
    else:
        form = DocumentForm()  # An empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()
    document = documents.last
    # Render list page with the documents and the form
    context = {'documents': documents, 'document': document, 'form': form}

    return render(request, 'list.html', context)
