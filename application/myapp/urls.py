from django.urls import path
from .views import *

urlpatterns = [
    path('', my_view, name='my-view'),
    path('keras_model_a_result', analyze_with_keras_model_a, name='analyze_with_keras_model_a'),
    path('keras_model_b_result', analyze_with_keras_model_b, name='analyze_with_keras_model_b'),
    path('keras_model_c_result', analyze_with_keras_model_c, name='analyze_with_keras_model_c'),
    path('keras_model_d_result', analyze_with_keras_model_d, name='analyze_with_keras_model_d'),

    path('cpplibrary_model_a_result', analyze_with_cpplibrary_model_a, name='analyze_with_cpplibrary_model_a'),
    path('img_not_found', img_not_found, name='img_not_found'),
    path('error', error, name='error')
]
