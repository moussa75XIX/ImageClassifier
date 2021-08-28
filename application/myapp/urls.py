from django.urls import path
from .views import *

urlpatterns = [
    path('', my_view, name='my-view'),
    path('result', analyze_with_keras_model_a, name='analyze_with_keras_model_a'),
    path('result', analyze_with_keras_model_b, name='analyze_with_keras_model_b'),
    path('result', analyze_with_keras_model_c, name='analyze_with_keras_model_c'),
    path('result', analyze_with_keras_model_d, name='analyze_with_keras_model_d'),
    path('img_not_found', img_not_found, name='img_not_found'),
    path('error', error, name='error')
]
