from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from .util import  predict_cerf
from .flair_util import dectect_info , sen_dectect_info
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
    
@csrf_exempt
def linggle_call(request):
    if request.method == "POST":
        if request.body:
            query = request.body.decode('utf-8').replace(',','').replace('.','').strip()
            print("input:", query)
            import requests
            API_URL = 'http://linggle.com/query/{}'
            def linggleit(query):
                r = requests.get(API_URL.format(query))
                return r.json()
            response = linggleit(query)
            if response['ngrams']:
                return JsonResponse(response, safe=False)
            else:
                token = query.split(' ')[-1]
                query = query.replace(token,'').strip()
                response = linggleit(query)
                return JsonResponse(response, safe=False)
                
            print(response)
            #return HttpResponse(cerf_level)
            #return JsonResponse(response, safe=False)
        else:
            return HttpResponse("Your request body is empty.")
    else:
        return HttpResponse("Please use POST.")

    

@csrf_exempt
def flair_dectect(request):
    if request.method == "POST":
        if request.body:
            print(request.body)
            json_data = json.loads(request.body)
            print(json_data)
            sentences = json_data["courpus"]
            print("input:", sentences)
            info = dectect_info(sentences)
            print("response:", info)
            return JsonResponse(info)
        else:
            return HttpResponse("Your request body is empty.")
    else:
        return HttpResponse("Please use POST.")
    
@csrf_exempt
def flair_sen_dectect(request):
    if request.method == "POST":
        if request.body:
            print(request.body)
            json_data = json.loads(request.body)
            print(json_data)
            sentences = json_data["courpus"]
            print("input:", sentences)
            info = dectect_info(sentences)
            print("response:", info)
            return JsonResponse(info)
        else:
            return HttpResponse("Your request body is empty.")
    else:
        return HttpResponse("Please use POST.")
            
        
    


    
#@csrf_exempt 
#def aes_dectect(request):
#    if request.method == "POST":
        # if request.POST:
            # json_data = request.POST
#        if request.body:
#            print(request.body)
#            json_data = json.loads(request.body)
#            print(json_data)
#            sentences = json_data["courpus"]
#            print("input:", sentences)
#            info = custom_dectect_info(sentences)
            #return HttpResponse(info)
#            return JsonResponse(info, safe=False)
#        else:
#            return HttpResponse("Your request body is empty.")
#    else:
#        return HttpResponse("Please use POST.")