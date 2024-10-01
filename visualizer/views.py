from django.shortcuts import render
from django.http import HttpResponsePermanentRedirect
from django.urls import reverse
from .data import SisFall
from .forms import UserForm

def dashboard(request):
    if request.method == 'GET':
        form = UserForm(request.GET)
        if form.is_valid():
            cd = form.cleaned_data
            return HttpResponsePermanentRedirect(reverse("visualizer:show_data",
                                                            args=(
                                                                cd["subject"],
                                                                cd["code"]
                                                            )
                                                )      )
    else:
        form = UserForm()
    
    context = {"form": form}
    return render(request, 'visualizer/dashboard.html', context)

def show_data(request, subject, code):
    sisfall = SisFall(subject, code)
    out = sisfall.read()
    trials = len(out)

    if request.method == 'GET' and request.GET.get("trial"):
        chosen = request.GET.get("trial")
        to_display = chosen
        to_data = out[chosen]
    else:
        if out:
            to_display, to_data = next(iter(out.items()))
        else:
            to_display, to_data = None, None
    
    context = {
        "data": out,
        "trials": trials,
        "to_display": to_display,
        "to_data": to_data,
        "subject": subject,
        "code": code,
    }
    return render(request, 'visualizer/displaydata.html', context)