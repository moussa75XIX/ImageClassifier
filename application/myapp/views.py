import glob
from django.shortcuts import redirect, render
from .models import Document
from .forms import DocumentForm
import ctypes
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute().parent.absolute()))
from call_native import main_linear_model


PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary.dll"

MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)

CLASSES = ["espagne", "france", "japon"]

espagne_file_model = "..\\saves\\linear_model\\train_linear_model_espagne_14_08_2021_H19_M21_S58.json"
france_file_model = "..\\saves\\linear_model\\train_linear_model_france_14_08_2021_H19_M21_S58.json"
japon_file_model = "..\\saves\\linear_model\\train_linear_model_japon_14_08_2021_H19_M21_S58.json"


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
        image_class = main_linear_model.get_classe(folder, image_file, espagne_file_model, france_file_model,
                                                   japon_file_model)
        try:
            context = {'documents': documents, "file_name": file.split('\\')[-1], 'document': document,
                       'result': result, 'predict': image_class[0]}
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
