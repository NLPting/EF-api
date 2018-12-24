from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from .util import  predict_cerf
import json

# Create your views here.
@csrf_exempt 
def aes_score(request):
    if request.method == "POST":
        # if request.POST:
            # json_data = request.POST
        if request.body:
            print(request.body)
            json_data = json.loads(request.body)
            print(json_data)
            sentences = json_data["courpus"]
            cerf_level = predict_cerf(sentences)
            print("input:", sentences)
            response = cerf_level
            print("response:", cerf_level)
            return HttpResponse(cerf_level)
            # return JsonResponse(response, safe=False)
        else:
            return HttpResponse("Your request body is empty.")
    else:
        return HttpResponse("Please use POST.")