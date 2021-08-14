from django.urls import path
from .views import my_view, analyze,img_not_found,error

urlpatterns = [
    path('', my_view, name='my-view'),
    path('result', analyze, name='analyze'),
    path('img_not_found', img_not_found, name='img_not_found'),
    path('error', error, name='error')
]
