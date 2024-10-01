from django.urls import path
from . import views

app_name = "visualizer"

urlpatterns = [
    path('dashboard/', views.dashboard, name="dashboard"),
    path('dashboard/<subject>/<code>', views.show_data, name="show_data")
]