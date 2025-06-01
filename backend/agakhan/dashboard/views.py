import PIL.Image
from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework.renderers import TemplateHTMLRenderer, JSONRenderer
from rest_framework.decorators import action
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticatedOrReadOnly, AllowAny, IsAuthenticated
from dashboard.models import Patient, Practitioner, MedicalScan, ClinicalResult, Branch, CustomUser, Examination, LabResults, Prediction, Report, SCAN_TYPE, ECG
from dashboard.serializers import PractitionerSerializer, PatientSerializer, BranchSerializer,ClinicalResultSerializer, MedicalScanSerializer, UserSerializer, LabResultsSerializer, ExaminationSerializer, PredictionSerializer, ReportSerializer, ECGSerializer, PredTypeSerializer, OTPVerifySerializer, SignInSerializer, ResendOTPSerializer
from django.shortcuts import get_object_or_404
from random import randint
from django.utils import timezone
import requests
from django.contrib.auth import login, logout, authenticate
from django.urls import reverse
from django.shortcuts import redirect
import uuid
import json
from agakhan.settings import TIARA_CONNECT_API_KEY
import secrets
from django.db import transaction
from django.db.utils import IntegrityError
from django.template import loader
import numpy as np
import wfdb
import tensorflow as tf
import joblib
import PIL
import cv2, os
import pandas as pd
from django.conf import settings
from urllib.parse import urlparse
from django.db.models import Avg
from datetime import date

# from django.contrib.auth import authenticate
# Create your views here.

class PractitionerViewset(ModelViewSet):
    queryset = Practitioner.objects.all()
    permission_classes = [IsAuthenticated]
    serializer_class = PractitionerSerializer

    @action(methods=['get', 'post'], detail=False, url_name='signin', url_path='signin.html', renderer_classes = [TemplateHTMLRenderer], permission_classes=[AllowAny])
    def signin(self, request):
        branches = Branch.objects.all()
        context = {'branches': branches}

        if request.method == 'POST':
            print(request.data)
            serializer = SignInSerializer(data=request.data)
            print(request)
            
            if not serializer.is_valid():
                return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST, template_name='signin.html')

        
            password = serializer.validated_data['password']
            user = serializer.validated_data['user']
            branch = serializer.validated_data['branch']
            branch = get_object_or_404(Branch, id=branch)
            
            print(user.serial_no)
            user_auth = authenticate(request, serial_no=user.serial_no, password=password, branch=branch)

            print(user.serial_no)
            
            
            login(request, user)
            phone_number = user.phone

            try:
                with transaction.atomic():
                    pract_user = get_object_or_404(Practitioner, user=user)
                    otp = str(secrets.randbelow(900000) + 100000)
                    user_auth = get_object_or_404(CustomUser, serial_no=user.serial_no)
                    user_auth.branch = branch
                    user_auth.save()
                    pract_user.otp = otp
                    print(otp)
                    pract_user.created_at = timezone.now()
                    pract_user.is_verified = False
                    
                    if timezone.now() - pract_user.trial_start > timezone.timedelta(hours=6):
                        pract_user.trial_start = timezone.now()
                        pract_user.trial_counter = 0
                    pract_user.save()
                    print(pract_user)

                    context['serial_no'] = user.serial_no
                    context['phone_number'] = user.phone
                    context['user'] = pract_user
                    print(user.phone)

                    if pract_user.trial_counter < 5 and timezone.now() - pract_user.trial_start < timezone.timedelta(hours=6):
                        ref_id = str(uuid.uuid4())

                        payload = {
                            "from": "TIARACONECT",
                            "to": phone_number,
                            "message": f"Your OTP is {otp}",
                            "refId": ref_id,
                            "messageType": "1",
                        }
                        headers = {
                            'Authorization': f'Bearer {TIARA_CONNECT_API_KEY}',
                            'Content-Type': 'application/json',
                        }
                        print(pract_user.user.serial_no)

                        response = requests.post(
                            'https://api2.tiaraconnect.io/api/messaging/sendsms',
                            json=payload,
                            headers=headers
                        )

                        if response.status_code == 200:
                            print(response)
                            pract_user.trial_counter += 1
                            pract_user.save()
                            context['message'] = 'OTP sent successfully.'
                            return Response(context, template_name='otp_verify.html', status=status.HTTP_200_OK)

                        return Response({'error': 'Failed to send OTP. Please try again.'},
                                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                        template_name='signin.html')

            except Exception as e:
                # Optional logging here
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, template_name='signup.html')
            return Response({'error': 'Check your credentials!'}, template_name='signin.html',status=status.HTTP_401_UNAUTHORIZED)
        return Response({'branches': branches}, status=status.HTTP_200_OK, template_name='signin.html')
    
    @action(methods=['get'], detail=False, url_name='signout', renderer_classes = [TemplateHTMLRenderer])
    def signout(self, request):
        logout(request)
        return Response(status=status.HTTP_200_OK, template_name='signin.html')
        
    @action(methods=['post'], detail=False, url_name='resend_otp', renderer_classes = [TemplateHTMLRenderer],)
    def resend_otp(self, request):
        context = {}
        serializer = ResendOTPSerializer(data=request.data)

        if not serializer.is_valid():
            print(serializer.error_messages)
            return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST, template_name='otp_verify.html')
        
        # fetch the user's data
        user = serializer.validated_data['user']
        phone_number = serializer.validated_data['phone_number']

        context['user'] = user
        context['phone_number'] = phone_number
        print(user)


        # generate a new otp for the user
        otp = str(secrets.randbelow(900000)+100000)
        try:
            with transaction.atomic():
                pract_user = get_object_or_404(Practitioner, user=user.serial_no)
                # set the new otp and send it to the user
                pract_user.otp = otp
                pract_user.created_at = timezone.now()
                pract_user.is_verified = False
                
                if timezone.now - pract_user.trial_start > timezone.timedelta(hours=6):
                    pract_user.trial_start = timezone.now()
                    pract_user.trial_counter = 0
                pract_user.save()


                context['user'] = pract_user
                context['phone_number'] = phone_number
                context['serial_no'] = pract_user.user.serial_no

                if pract_user.trial_counter < 5 and timezone.now - pract_user.trial_start < timezone.timedelta(hours=6):
                    ref_id = str(uuid.uuid4())
                    payload = {
                        "from": "TIARACONECT",
                        "to": phone_number,
                        "message": f"Your OTP is {otp}",
                    }
                    headers = {
                        'Authorization': f'Bearer {TIARA_CONNECT_API_KEY}',
                        'Content-Type': 'application/json',
                    }

                    response = requests.post(
                        'https://api2.tiaraconnect.io/api/messaging/sendsms',
                        json=payload,
                        headers=headers
                    )



                    if response.status_code == 200:
                        # redirect to the same page
                        context['message'] = 'OTP sent successfully.'
                        pract_user.trial_counter += 1
                        pract_user.save()
                        return Response(context, template_name='otp_verify.html', status=status.HTTP_200_OK)
                    context['error'] = 'Failed to send OTP. Please try again.'
                    return Response(context,status=status.HTTP_500_INTERNAL_SERVER_ERROR, template_name='otp_verify.html')
                
                context['error'] = 'Failed to send OTP. You are only allowed to request OTP five times in six hours duration.'
                return Response(context, status=status.HTTP_403_FORBIDDEN, template_name='otp_verify.html')

        except Exception as e:
            print(e)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, template_name='otp_verify.html')
    

    @action(methods=['get', 'post'], detail=False, url_name='otp_verify', url_path='otp_verify.html', renderer_classes = [TemplateHTMLRenderer])
    def otp_verify(self, request):
        if request.method == 'GET':
            return Response(status=status.HTTP_200_OK, template_name='otp_verify.html')

        print(request.data)
        if request.method == 'POST':
            serializer = OTPVerifySerializer(data=request.data)

            if not serializer.is_valid():
                print(serializer.error_messages)
                print(serializer._errors)
                print(serializer.errors)
                return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST, template_name='otp_verify.html')
            
            otp = serializer.validated_data['otp']
            user = serializer.validated_data['user']
            print(user.serial_no)
            try:
                pract_user = get_object_or_404(Practitioner, user=user, otp=otp)
                print(pract_user)
                if timezone.now() - pract_user.created_at > timezone.timedelta(minutes=15):
                    return Response({'error': 'OTP has expired.'}, status=status.HTTP_403_FORBIDDEN, template_name='otp_verify.html')

                
                # Verify user and clear OTP
                print(pract_user)
                pract_user.verified = True
                pract_user.otp = 0
                pract_user.created_at = timezone.now()
                pract_user.save()
                print(pract_user)
                serial_no = user.serial_no
                print(serial_no)
                user = get_object_or_404(CustomUser, serial_no=serial_no)
                print(user)
                login(request, user)
                return Response({'message': 'OTP verified successfully.', 'user': user}, status=status.HTTP_200_OK, template_name='home.html')

            except CustomUser.DoesNotExist:
                return Response({'error': 'Invalid OTP or phone number.'}, status=status.HTTP_204_NO_CONTENT, template_name='otp_verify.html')
            except Exception as e:
                return Response({'error': e}, status=status.HTTP_404_NOT_FOUND, template_name='otp_verify.html')

    @action(methods=['get', 'post'], detail=False, url_name='signup', url_path='signup.html', permission_classes=[AllowAny], renderer_classes = [TemplateHTMLRenderer])
    def signup(self, request):
        from dashboard.models import GENDER_CHOICES
        branches = Branch.objects.all().order_by('name')  # Get all branches
        gender = GENDER_CHOICES[:]
        context = {'branches': branches, 'gender': gender}
        # print(request.data)

        if request.method == 'POST':
            serializer = UserSerializer(data=request.data)
            if not serializer.is_valid():
                print(serializer.error_messages)
                print(serializer._errors)
                return Response({'errors': serializer.errors},template_name='signup.html')

            phone_number = serializer.validated_data['phone']
            serial_no = serializer.validated_data['serial_no']
            try:
                with transaction.atomic():
                    user = serializer.save()
                    # logout(request.user)
                    practitioner_user = Practitioner.objects.create(user=user, otp = str(secrets.randbelow(900000) + 100000), created_at=timezone.now())
                    # practitioner_user.save()
                    # logout(request.user)
                    login(request, user)

                    context['serial_no'] = practitioner_user.user.serial_no
                    context['phone_number'] = practitioner_user.user.phone
                    context['user'] = practitioner_user
                    # request.session['otp_data'] = context
                    # request.session['from_user'] = user.serial_no
                    ref_id = str(uuid.uuid4())
                    print(practitioner_user)

                    payload = {
                        "from": "TIARACONECT",
                        "to": phone_number,
                        "message": f"Your OTP is {practitioner_user.otp}",
                        "refId": ref_id,
                        "messageType": "1",
                    }
                    headers = {
                        'Authorization': f'Bearer {TIARA_CONNECT_API_KEY}',
                        'Content-Type': 'application/json',
                    }

                    response = requests.post(
                        'https://api2.tiaraconnect.io/api/messaging/sendsms',
                        json=payload,
                        headers=headers
                    )
                    code = int(response.status_code)
                    print(type(code))
                    print(phone_number)

                    

                    if code == 200:
                        print(request.session.get('from_user'))
                        print(phone_number, serial_no)
                        practitioner_user.trial_counter = 1
                        practitioner_user.save()
                        return Response(context, template_name='otp_verify.html', status=status.HTTP_200_OK)
                    else:
                        return Response({'error': 'Failed to send OTP. Please try again.', **context}, template_name='otp_request.html', status=status.HTTP_403_FORBIDDEN)

            except Exception as e:
                print(e)
                return Response({'error is': str(e), "context": context}, template_name='signin.html', status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(context, template_name='signup.html', status=status.HTTP_200_OK)
    
    @action(methods=['get'], detail=False, url_name='home', url_path='home.html', renderer_classes = [TemplateHTMLRenderer])
    def home(self, request):
        return Response({'user': request.user}, status=status.HTTP_200_OK, template_name='home.html')

class RegisterPatientViewSet(ModelViewSet):
    queryset = Patient.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = PatientSerializer

    @action(methods=['get', 'post'], detail=False, url_name='register_patient', url_path='register_patient.html', renderer_classes = [TemplateHTMLRenderer])
    def register_patient(self, request):
        branch = request.session.get('branch')
        # branch = request.user.branch
        print(branch)
        if request.method == 'POST':
            serializer = PatientSerializer(data=request.data)
            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='register_patient.html')
            
            patient = serializer.save()

            return Response({'patient': patient, 'next_step': 'exam-examination', 'next_name': 'Proceed for Examination', 'message': "Patien's details successfully recorded"}, status=status.HTTP_201_CREATED, template_name='confirmation.html')
        return Response({'branch': branch}, status=status.HTTP_200_OK, template_name='register_patient.html')

class ExaminationViewSet(ModelViewSet):
    queryset = Examination.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = ExaminationSerializer

    @action(methods=['get', 'post'], detail=False, url_name='examination', url_path='examination.html', renderer_classes = [TemplateHTMLRenderer])
    def examination(self, request):
        if request.method == 'POST':
            serializer = ExaminationSerializer(data=request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='examination.html')
            
            patient = serializer.save()

            return Response({'patient': patient, 'next_step': 'clinic-clinical_assessment', 'next_name': 'Proceed for Clinical Assessment', 'message': 'Examination results recorded successfully.'}, status=status.HTTP_201_CREATED, template_name='confirmation.html')     
        if request.method == 'GET':  
            patient = request.session.get('patient')
            # patient = request.data
            print(patient)    
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='examination.html')


class ClinicalResultViewSet(ModelViewSet):
    queryset = ClinicalResult.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = ClinicalResultSerializer

    @action(methods=['get', 'post'], detail=False, url_name='clinical_assessment', url_path='clinical_assessment.html', renderer_classes = [TemplateHTMLRenderer])
    def clinical_assessment(self, request):
        if request.method == 'POST':
            serializer = ClinicalResultSerializer(data=request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='clinical_assessment.html')
            
            patient = serializer.save()

            return Response({'patient': patient, 'next_step': 'lab-lab_results', 'next_name': 'Proceed for Recording Lab Results', 'message': 'Clinical Assessment results recorded successfully.'}, status=status.HTTP_201_CREATED, template_name='confirmation.html')  
        if request.method == 'GET':      
            patient = request.session.get('patient')    
            patient = request.session.get('patient')
            # patient = request.data
            print(patient) 
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='clinical_assessment.html')

class LabResultsViewSet(ModelViewSet):
    queryset = LabResults.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = LabResultsSerializer

    @action(methods=['get', 'post'], detail=False, url_name='lab_results', url_path='lab_results.html', renderer_classes = [TemplateHTMLRenderer])
    def lab_results(self, request):
        if request.method == 'POST':
            serializer = ClinicalResultSerializer(data=request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='lab_results.html')
            
            patient = serializer.save()

            return Response({'patient': patient, 'next_step': 'lab-predict', 'next_name': 'Predict Diagnosis', 'message': 'Lab results recorded successfully.'}, status=status.HTTP_201_CREATED, template_name='confirmation.html')      
        if request.method == 'GET':
            patient = request.session.get('patient')   
            # patient = request.data
            print(patient)    
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='lab_results.html')

    @action(methods=['get', 'post'], detail=False, url_name='predict', url_path='predict.html', renderer_classes = [TemplateHTMLRenderer, JSONRenderer])
    def predict(self, request):
        # load the model and preprocessor
        model_path = os.path.join(settings.BASE_DIR, 'models', 'dnn_base_exam.h5')
        norm_model = tf.keras.models.load_model(model_path)
        risk_model_extra = joblib.load(os.path.join(settings.BASE_DIR, 'models', 'xgb_base.pkl'))
        preprocessor = joblib.load('models/heart_preprocessor.pkl')
        if request.method == 'POST':
            serializer = PredTypeSerializer(data=request.data)

            if not serializer.is_valid():
                print(serializer.error_messages)
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED)
            
            serial_no = serializer.validated_data['serial_no']
            print(serial_no)
            patient = get_object_or_404(Patient, serial_no=serial_no)
            print(patient)
            lab = get_object_or_404(LabResults, patient=serial_no)
            print(lab)
            clinic = get_object_or_404(ClinicalResult, patient=serial_no)
            print(clinic)
            exam = get_object_or_404(Examination, patient=serial_no)
            print(exam)
            dob = patient.dob 
            print(dob) # Convert to `date` object if it's a datetime
            clinic_date = lab.date  # Same here

            print(clinic_date)
            age = clinic_date.year - dob.year - ((clinic_date.month, clinic_date.day) < (dob.month, dob.day))
            gender = patient.gender
            bp = exam.bp
            chol = lab.chol_level
            ex_habits = exam.ex_habits
            smoking = exam.smoking_habits
            fam = exam.family_history
            diab = clinic.diabetes
            bmi = clinic.bmi
            hpb = clinic.hbp
            low_hdl = lab.low_hdl
            high_ldl = lab.high_ldl
            alc = exam.alc_habits
            stress = exam.stress_levels
            sleep = exam.avrg_sleep
            sugar = exam.sugar_cons
            trig = lab.trig_level
            fbs = lab.fasting_blood_sugar
            crp = lab.crp_level
            homo = lab.homo_level
            weight = exam.weight
            abdominal_circ = exam.abdominal_circ
            height = exam.height
            hdl_reading = lab.hdl_reading
            # attributes for predictions to be successful predictions
            columns = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
                        'Exercise Habits', 'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI',
                        'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
                        'Alcohol Consumption', 'Stress Level', 'Sleep Hours',
                        'Sugar Consumption', 'Triglyceride Level', 'Fasting Blood Sugar',
                        'CRP Level', 'Homocysteine Level']
            data_df = pd.DataFrame([[age, gender, bp, chol, ex_habits, smoking, fam, diab, bmi, hpb, low_hdl, high_ldl, alc, stress,sleep, sugar, trig, fbs, crp, homo]], columns=columns)
            column_risk = ['SEX', 'AGE', 'WEIGHT', 'HEIGHT', 'BMI', 'ABDOMINAL CIRCUMFERENCE', 'BLOOD PRESSURE', 'TOTAL CHOLESTEROL', 'HDL', 'FASTING BLOOD SUGAR', 'SMOKING']
            gender = 1 if gender == 'M' else 0
            smoking = 1 if smoking == 'Yes' else 0
            data_risk = pd.DataFrame([[gender, age, weight, height, bmi, abdominal_circ, bp, chol, hdl_reading, fbs, smoking]], columns=column_risk)
            
            # preprocess the data and make predictions
            data_df = preprocessor.transform(data_df)
            probability = norm_model.predict(data_df)
            risk_value = risk_model_extra.predict(data_risk)
            risk_conf_score = np.max(risk_model_extra.predict_proba(data_risk))

            risk_map = {'LOW': 0, 'INTERMEDIARY': 1, 'HIGH': 2}
            risk_class = [key for key in  risk_map.keys() if risk_map[key] == risk_value][0]

            pred_class = int(probability >= 0.5)
            # classes = ['No', 'Yes']
            class_name = 'No' if pred_class == 0 else 'Yes'
            conf_score = np.round(probability if pred_class == 1 else 1 - probability, 4)

            pred = Prediction.objects.create(patient=patient, prediction_type='lab', confidence_score= conf_score*100, predicted_class = pred_class, predicted_name=class_name, classes_probabilities=f'[{probability*100}, {(1 - probability)*100}]', risk_class=risk_map[risk_class], disease_class=pred_class, date=timezone.now(), risk_conf_score=risk_conf_score*100)
            print(pred.predicted_name)

            predictions = Prediction.objects.filter(patient=patient).order_by('-date')
            
            # get the last visit or predictions made in the last visit
            last_ecg = ECG.objects.filter(patient=patient).order_by('-date').first()
            last_lab = LabResults.objects.filter(patient=patient).order_by('-date').first()
            last_chest = MedicalScan.objects.filter(patient=patient).order_by('-date').first()
            # Calculate age
            today = date.today()
            age = today.year - patient.dob.year - (
                (today.month, today.day) < (patient.dob.month, patient.dob.day))
            
                        
            # Get latest dates
            latest_ecg = predictions.filter(prediction_type='ecg').first()
            latest_lab = predictions.filter(prediction_type='lab').first()
            
            context = {
                'patient': PatientSerializer(patient).data,
                'pred': PredictionSerializer(pred).data,
                'predictions': PredictionSerializer(predictions, many=True).data,
                'age': age,
                'lab_count': predictions.filter(prediction_type='lab').count(),
                'ecg_count': predictions.filter(prediction_type='ecg').count(),
                'chest_count': predictions.filter(prediction_type='chest').count(),
                'avg_confidence': predictions.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0,
                'high_risk_count': predictions.filter(risk_class=True).count(),
                'latest_ecg_date': last_ecg.date if last_ecg else "N/A",
                'latest_lab_date': last_lab.date if last_lab else "N/A",
                }

            # Return JSON with redirect info for API calls
            if request.accepted_renderer.format == 'json':
                context['patient_id'] = patient.serial_no
                return Response({
                    'redirect_url': (
                        reverse('report-reports')
                        + f'?patient_id={patient.serial_no}'
                    ),
                    'context': context
                }, status=status.HTTP_201_CREATED)
            
            # Return HTML template for direct browser access
            return Response(context, template_name='report.html', status=status.HTTP_201_CREATED)
        if request.method == 'GET':
            # Fetch the availabe patients
            patient = Patient.objects.all()
            
            return Response({'patient': patient}, template_name='predict.html', status=status.HTTP_200_OK)
            

class ECGViewSet(ModelViewSet):
    queryset = ECG.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = ECGSerializer

    @action(methods=['get', 'post'], detail=False, url_name='ecg', url_path='ecg.html', renderer_classes = [TemplateHTMLRenderer])
    def ecg(self, request):
        if request.method == 'POST':
            serializer = ClinicalResultSerializer(data = request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='ecg.html')
            
            patient = serializer.save()
            return Response({'patient': patient, 'next_step': 'ecg-predict', 'next_name': 'Predict Diagnosis', 'message': 'ECG files recorded successfully.'}, status=status.HTTP_201_CREATED, template_name='confirmation.html')      
        if request.method == 'GET':
            patient = request.session.get('patient')
            # patient = request.data
            print(patient)     
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='ecg.html')
        
    @action(methods=['get', 'post'], detail=False, url_name='predict', url_path='predict.html', renderer_classes = [TemplateHTMLRenderer, JSONRenderer])
    def predict(self, request):
        model_path = os.path.join(settings.BASE_DIR, 'models', 'cnn_ecg_deep.keras')
        ecg_model = tf.keras.models.load_model(model_path) #import the ecg model
        if request.method == 'POST':
            print('..................................hhhhhhhh....................')
            serializer = PredTypeSerializer(data= request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='predict.html')
            
            # patient = serializer.save()
            # scan_done = serializer.validated_data['pred_type']
            patient = serializer.validated_data['serial_no']
            patient = get_object_or_404(Patient, serial_no=patient)
            content_scan = get_object_or_404(ECG, patient=patient)

            # Step 1: Extract the path from the URL
            url = content_scan.scan_hea
            print(url)

            # Step 2: Get the filename (with extension)
            filename_with_ext = os.path.basename(str(url))  # "00001_lr.hea"

            # Step 3: Get the filename without extension (needed by wfdb)
            file_id = os.path.splitext(filename_with_ext)[0]  # "00001_lr"

            # Step 4: Construct local path (assuming you store data locally)
            local_path = os.path.join('data/scans/ecg', file_id)
            print(local_path)

            print(local_path)
            sig, attr = wfdb.rdsamp(local_path)
            prediction_probablities = ecg_model.predict(np.expand_dims(sig, axis=0))[0]
            prediction_class = np.argmax(prediction_probablities)
            ecg_cols = ['HYP', 'NORM', 'CD', 'MI', 'STTC', 'NN']
            predicted_class = np.argmax(ecg_model.predict(np.expand_dims(sig, axis=0)))
            print(f'The patient has disease belonging the class {ecg_cols[predicted_class]}')
            np.set_printoptions(suppress=True) 
            probs = np.round(prediction_probablities, 4) * 100
            print(np.round(prediction_probablities, 4) * 100)
            
            conf_score = round(prediction_probablities[predicted_class] * 100, 4)
            class_name = ecg_cols[predicted_class]
            risk_class = 0 if class_name == 'NORM' else 1
            dis_diag = predicted_class
            pred = Prediction.objects.create(patient=patient, confidence_score=conf_score, predicted_class=predicted_class, predicted_name=class_name, classes_probabilities=probs.tolist(), risk_class=risk_class, disease_class=dis_diag)

            predictions = Prediction.objects.filter(patient=patient).order_by('-date')
            
            # get the last visit or predictions made in the last visit
            last_ecg = ECG.objects.filter(patient=patient).order_by('-date').first()
            last_lab = LabResults.objects.filter(patient=patient).order_by('-date').first()
            last_chest = MedicalScan.objects.filter(patient=patient).order_by('-date').first()
            # Calculate age
            today = date.today()
            age = today.year - patient.dob.year - (
                (today.month, today.day) < (patient.dob.month, patient.dob.day))
            
                        
            # Get latest dates
            latest_ecg = predictions.filter(prediction_type='ecg').first()
            latest_lab = predictions.filter(prediction_type='lab').first()
            
            context = {
                'patient': PatientSerializer(patient).data,
                'pred': PredictionSerializer(pred).data,
                'predictions': PredictionSerializer(predictions, many=True).data,
                'age': age,
                'lab_count': predictions.filter(prediction_type='lab').count(),
                'ecg_count': predictions.filter(prediction_type='ecg').count(),
                'chest_count': predictions.filter(prediction_type='chest').count(),
                'avg_confidence': predictions.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0,
                'high_risk_count': predictions.filter(risk_class=True).count(),
                'latest_ecg_date': last_ecg.date if last_ecg else "N/A",
                'latest_lab_date': last_lab.date if last_lab else "N/A",
                }

            # Return JSON with redirect info for API calls
            if request.accepted_renderer.format == 'json':
                context['patient_id'] = patient.serial_no
                return Response({
                    'redirect_url': (
                        reverse('report-reports')
                        + f'?patient_id={patient.serial_no}'
                    ),
                    'context': context
                }, status=status.HTTP_201_CREATED)      
        if request.method == 'GET':
            patient = Patient.objects.all()
            # patient = request.data
            print(patient)     
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='predict.html')


class MedicalScanViewSet(ModelViewSet):
    queryset = MedicalScan.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = MedicalScanSerializer

    @action(methods=['get','post'], detail=False, url_name='medical_scan', url_path='medical_scan.html', renderer_classes = [TemplateHTMLRenderer])
    def medical_scan(self, request):
        if request.method == 'POST':
            serializer = MedicalScanSerializer(data=request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='medical_scan.html')
            
            patient = serializer.save()
            return Response({'patient': patient}, status=status.HTTP_201_CREATED, template_name='prediction.html')           
        if request.method == 'GET':
            # patient = get_object_or_404(Patient, serial_no=request)
            patient = request.data
            return Response({'patient': patient, 'next_step': 'scan-predict', 'next_name': 'Predict Diagnosis', 'message': 'Chest X-ray recorded successfully.'}, status=status.HTTP_200_OK, template_name='medical_scan.html')
        
    @action(methods=['get', 'post'], detail=False, url_name='predict', url_path='predict.html', renderer_classes = [TemplateHTMLRenderer, JSONRenderer])
    def predict(self, request):
        model_path = os.path.join(settings.BASE_DIR, 'models', 'best_model_8.keras')
        chest_model = tf.keras.models.load_model(model_path) # load chest model
        if request.method == 'POST':
            serializer = PredTypeSerializer(data=request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='predict.html')
            

            patient = serializer.validated_data['serial_no']
            patient = get_object_or_404(Patient, serial_no=patient)
            content_scan = MedicalScan.objects.filter(patient=patient).order_by('-date').first()
            print(content_scan)
            # Step 1: Extract the path from the URL
            url = content_scan.scan_content
            url = os.path.join(settings.BASE_DIR, str(url))
            print(url)

            print(url)
            image = cv2.imread((url))
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = image / 255.0
            print(image.shape)

            prediction_probablities = chest_model.predict(np.expand_dims(image, axis=0))[0]
            prediction_class = np.argmax(prediction_probablities)
            final_classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
                'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
                'Infiltration', 'Mass', 'No Finding', 'Nodule',
                'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
            # index = prediction_probablities.get_index(prediction_class)
            class_name = final_classes[prediction_class]
            risk_class = 0 if class_name == 'No Finding' else 1
            dis_diag = prediction_class
            conf_score = round(prediction_probablities[prediction_class] * 100, 4)
            print(class_name)
            print(conf_score)
            np.set_printoptions(suppress=True) 
            probs = np.round(prediction_probablities * 100, 2)
            print(probs )
            print()

            pred = Prediction.objects.create(patient=patient, confidence_score=conf_score, predicted_class=prediction_class, predicted_name=class_name, classes_probabilities=probs, risk_class=risk_class, disease_class=dis_diag, prediction_type='x_ray')


            predictions = Prediction.objects.filter(patient=patient).order_by('-date')
            
            # get the last visit or predictions made in the last visit
            last_ecg = ECG.objects.filter(patient=patient).order_by('-date').first()
            last_lab = LabResults.objects.filter(patient=patient).order_by('-date').first()
            last_chest = MedicalScan.objects.filter(patient=patient).order_by('-date').first()
            # Calculate age
            today = date.today()
            age = today.year - patient.dob.year - (
                (today.month, today.day) < (patient.dob.month, patient.dob.day))
            
                        
            # Get latest dates
            latest_ecg = predictions.filter(prediction_type='ecg').first()
            latest_lab = predictions.filter(prediction_type='lab').first()
            
            context = {
                'patient': PatientSerializer(patient).data,
                'pred': PredictionSerializer(pred).data,
                'predictions': PredictionSerializer(predictions, many=True).data,
                'age': age,
                'lab_count': predictions.filter(prediction_type='lab').count(),
                'ecg_count': predictions.filter(prediction_type='ecg').count(),
                'chest_count': predictions.filter(prediction_type='chest').count(),
                'avg_confidence': predictions.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0,
                'high_risk_count': predictions.filter(risk_class=True).count(),
                'latest_ecg_date': last_ecg.date if last_ecg else "N/A",
                'latest_lab_date': last_lab.date if last_lab else "N/A",
                }

            # Return JSON with redirect info for API calls
            if request.accepted_renderer.format == 'json':
                context['patient_id'] = patient.serial_no
                return Response({
                    'redirect_url': (
                        reverse('report-reports')
                        + f'?patient_id={patient.serial_no}'
                    ),
                    'context': context
                }, status=status.HTTP_201_CREATED)      
        if request.method == 'GET':
            patient = request.session.get('patient')
            # patient = request.data
            print(patient)     
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='predict.html')

class BranchViewSet(ModelViewSet):
    queryset = Branch.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = BranchSerializer

class PredictViewSet(ModelViewSet):
    queryset = Prediction.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = PredictionSerializer

    @action(methods=['get', 'post'], detail=False, url_name='prediction', url_path='prediction.html', renderer_classes = [TemplateHTMLRenderer])
    def prediction(self, request):
        if request.method == 'POST':
            serializer = PredictionSerializer(data=request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='prediction.html')
            
            patient = serializer.validated_data['serial_no']
            pred_type = serializer.validated_data['prediction_type']
            try:
                if pred_type == 'ecg':
                    data = get_object_or_404(ECG, patient=patient).objects.order_by('-date').first()
                    return Response({'data': data}, template_name=reverse('ecg-predict'))
                
                elif pred_type == 'lab_stage':
                    data = get_object_or_404(LabResults, patient=patient).objects.order_by('-created_by').first()
                    return Response({'data': data}, template_name=reverse('lab-predict'))
                
                else:
                    data = get_object_or_404(MedicalScan, patient=patient).objects.order_by('-created_at').first()
                    return Response({'data': data}, template_name=reverse('scan-predict'))
                
            except Exception as e:
                return Response({'patient': patient}, status=status.HTTP_404_NOT_FOUND, template_name='prediction.html')
            
            # return Response({'patient': patient}, status=status.HTTP_201_CREATED, template_name='reports.html')           
        if request.method == 'GET':
            patient = request.data
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='prediction.html')

class ReportViewSet(ModelViewSet):
    queryset = Report.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = ReportSerializer

    # @action(methods=['get', 'post'], detail=False, url_name='reports', url_path='reports.html', renderer_classes = [TemplateHTMLRenderer])
    # def reports(self, request):
    #     if request.method == 'GET':
    #         patient_id = request.query_params.get('patient_id')
    #         return Response({'patient': patient_id}, status=status.HTTP_200_OK, template_name='reports.html')
    #     return Response(status=status.HTTP_200_OK, template_name='reports.html')

    @action(methods=['get'], detail=False, 
            url_name='reports', url_path='reports.html', 
            renderer_classes=[TemplateHTMLRenderer])
    def reports(self, request):
        patient_id = request.query_params.get('patient_id')
        print(patient_id)
        if not patient_id:
            return Response({"error": "Patient ID is required"}, 
                           status=status.HTTP_400_BAD_REQUEST,
                           template_name='error.html')
        
        try:
            patient = Patient.objects.get(serial_no=patient_id)
        except Patient.DoesNotExist:
            return Response({"error": "Patient not found"}, 
                           status=status.HTTP_404_NOT_FOUND,
                           template_name='error.html')
        
        # Calculate age
        today = date.today()
        age = today.year - patient.dob.year - (
            (today.month, today.day) < (patient.dob.month, patient.dob.day))
        
        # Get predictions
        predictions = Prediction.objects.filter(patient=patient).order_by('-date')
        
        # Get latest dates
        latest_ecg = predictions.filter(prediction_type='ecg').first()
        latest_lab = predictions.filter(prediction_type='lab').first()

        avg_conf_ecg = predictions.filter(prediction_type='ecg').aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
        avg_conf_lab = predictions.filter(prediction_type='lab').aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
        avg_conf_x_ray = predictions.filter(prediction_type='x_ray').aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
        
        context = {
            'patient': PatientSerializer(patient).data,
            'predictions': predictions,
            'age': age,
            'lab_count': predictions.filter(prediction_type='lab').count(),
            'ecg_count': predictions.filter(prediction_type='ecg').count(),
            'chest_count': predictions.filter(prediction_type='chest').count(),
            'avg_confidence': (predictions.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0),
            'high_risk_count': predictions.filter(risk_class=True).count(),
            'latest_ecg_date': latest_ecg.date if latest_ecg else "N/A",
            'latest_lab_date': latest_lab.date if latest_lab else "N/A",
            'avg_score_ecg':avg_conf_ecg,
            'avg_score_x_ray':avg_conf_x_ray,
            'avg_score_lab':avg_conf_lab
        }
        
        return Response(context, template_name='reports.html')


