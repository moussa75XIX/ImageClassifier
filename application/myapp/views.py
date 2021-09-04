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

from scripts_notebooks import predict_keras_mlp_model, predict_cpplibrary_mlp_model

PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary.dll"

MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)

keras_model_a = tf.keras.models.load_model("../models/keras_models/model_a.h5")
keras_model_b = tf.keras.models.load_model("../models/keras_models/model_b.h5")
keras_model_c = tf.keras.models.load_model("../models/keras_models/model_c.h5")
keras_model_d = tf.keras.models.load_model("../models/keras_models/model_d.h5")

MY_LIB.load_mlp_model.argtypes = [ctypes.c_char_p]
MY_LIB.load_mlp_model.restype = ctypes.c_void_p
cpplibrary_model_a = MY_LIB.load_mlp_model("..\\models\\cpplibrary_models\\model_a.json".encode('utf-8'))


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


def analyze_with_keras_model_a(request):
    try:
        # Load documents for the list page
        documents = Document.objects.all()
        document = documents.last

        # Render list page with the documents and the form
        result = "Résultat de l'analyse"
        file = ""

        for files in glob.iglob('media\\documents\\**\\*.*', recursive=True):
            file = files

        image_class = predict_keras_mlp_model.predict(keras_model_a,CLASSES,CLASSES_SIZE,file)
        try:
            context = {'documents': documents, "file_name": file.split('\\')[-1], 'document': document,
                       'result': result, 'predict': image_class}
            try:
                shutil.rmtree("media\\documents")
            except:
                pass

            return render(request, 'keras_model_a_result.html', context)

        except:
            return redirect('error')
    except:
        return redirect('img_not_found')


def analyze_with_keras_model_b(request):
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

        image_class = predict_keras_mlp_model.predict(keras_model_b,CLASSES,CLASSES_SIZE,file)
        try:
            context = {'documents': documents, "file_name": file.split('\\')[-1], 'document': document,
                       'result': result, 'predict': image_class}
            try:
                shutil.rmtree("media\\documents")
            except:
                pass

            return render(request, 'keras_model_b_result.html', context)

        except:
            return redirect('error')
    except:
        return redirect('img_not_found')


def analyze_with_keras_model_c(request):
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

        image_class = predict_keras_mlp_model.predict(keras_model_c,CLASSES,CLASSES_SIZE,file)
        try:
            context = {'documents': documents, "file_name": file.split('\\')[-1], 'document': document,
                       'result': result, 'predict': image_class}
            try:
                shutil.rmtree("media\\documents")
            except:
                pass

            return render(request, 'keras_model_c_result.html', context)

        except:
            return redirect('error')
    except:
        return redirect('img_not_found')


def analyze_with_keras_model_d(request):
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

        image_class = predict_keras_mlp_model.predict(keras_model_d,CLASSES,CLASSES_SIZE,file)
        try:
            context = {'documents': documents, "file_name": file.split('\\')[-1], 'document': document,
                       'result': result, 'predict': image_class}
            try:
                shutil.rmtree("media\\documents")
            except:
                pass

            return render(request, 'keras_model_d_result.html', context)

        except:
            return redirect('error')
    except:
        return redirect('img_not_found')


def analyze_with_cpplibrary_model_a(request):
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

        image_class = predict_cpplibrary_mlp_model.predict(cpplibrary_model_a,CLASSES,CLASSES_SIZE,file)
        try:
            context = {'documents': documents, "file_name": file.split('\\')[-1], 'document': document,
                       'result': result, 'predict': image_class}
            try:
                shutil.rmtree("media\\documents")
            except:
                pass

            return render(request, 'cpplibrary_model_a_result.html', context)

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
