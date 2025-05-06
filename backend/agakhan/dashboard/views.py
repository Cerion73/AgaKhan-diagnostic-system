import PIL.Image
from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.decorators import action
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticatedOrReadOnly, AllowAny, IsAuthenticated
from dashboard.models import Patient, Practitioner, MedicalScan, ClinicalResult, Branch, CustomUser, Examination, LabResults, Prediction, Report, SCAN_TYPE, ECG
from dashboard.serializers import PractitionerSerializer, PatientSerializer, BranchSerializer,ClinicalResultSerializer, MedicalScanSerializer, UserSerializer, LabResultsSerializer, ExaminationSerializer, PredictionSerializer, ReportSerializer, ECGSerializer
from django.shortcuts import get_object_or_404
from random import randint
from django.utils import timezone
import requests
from django.contrib.auth import login, logout
from django.urls import reverse
from django.shortcuts import redirect
import uuid
import json
from agakhan.settings import TIARA_CONNECT_API_KEY
import secrets
from django.db import transaction
from django.db.utils import IntegrityError
import numpy as np
import wfdb
import tensorflow as tf
import joblib
import PIL
import cv2
import pandas as pd
# from django.contrib.auth import authenticate
# Create your views here.

class PractitionerViewset(ModelViewSet):
    queryset = Practitioner.objects.all()
    permission_classes = [AllowAny]
    serializer_class = PractitionerSerializer

    @action(methods=['get', 'post'], detail=False, url_name='signin', url_path='signin.html', renderer_classes = [TemplateHTMLRenderer])
    def signin(self, request):
        branches = Branch.objects.all()

        if request.method == 'POST':
            serializer = PractitionerSerializer(data=request.data)
            print(request.data)

            if serializer.is_valid():
                password = serializer.validated_data['password']
                serial_no = serializer.validated_data['serial_no']
                branch = serializer.validated_data['branch']
                
                user = get_object_or_404(CustomUser, serial_no=serial_no, password=password, branch=branch)
                # user = Practitioner.objects.get(serial_no=serial_no, password=password)
                # phone_number = user.phone
                print(user)
                try:
                    login(request, user)
                    return Response({'message': 'OTP sent successfully.'}, template_name='home.html')
                except Exception as e:
                    return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, template_name='signin.html')

            #     try:
            #         with transaction.atomic():
            #             otp = str(secrets.randbelow(900000) + 100000)
            #             user.otp = otp
            #             user.created_at = timezone.now()
            #             user.is_verified = False
            #             user.save()

            #             ref_id = str(uuid.uuid4())

            #             payload = {
            #                 "from": "TIARACONECT",
            #                 "to": phone_number,
            #                 "message": f"Your OTP is {otp}",
            #                 "refId": ref_id,
            #                 "messageType": "1",
            #             }
            #             headers = {
            #                 'Authorization': f'Bearer {TIARA_CONNECT_API_KEY}',
            #                 'Content-Type': 'application/json',
            #             }

            #             response = requests.post(
            #                 'https://api2.tiaraconnect.io/api/messaging/sendsms',
            #                 json=payload,
            #                 headers=headers
            #             )

            #             if response.status_code == 200:
            #                 return Response({'message': 'OTP sent successfully.'}, template_name='otp_verify.html')

            #             else:
            #                 return Response({'error': 'Failed to send OTP. Please try again.'},
            #                             status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            #                             template_name='otp_request.html')

            #     except Exception as e:
            #         # Optional logging here
            #         return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, template_name='signup.html')
            # return Response({'error': 'Check your credentials!'}, template_name='signin.html',status=status.HTTP_200_OK)
        return Response({'branches': branches}, status=status.HTTP_200_OK, template_name='signin.html')
        
    @action(methods=['get', 'post'], detail=False, url_name='otp_request', url_path='otp_request.html', renderer_classes = [TemplateHTMLRenderer])
    def otp_request(self, request):
        if request.method == 'POST':
            serializer = PractitionerSerializer(data=request.data)

            if serializer.is_valid():
                phone_number = serializer.validated_data['phone']
                otp = str(secrets.randbelow(900000)+100000)
                user = Practitioner.objects.get(user=request.user)
                try:
                    with transaction.atomic():
                        user.otp = otp
                        user.created_at = timezone.now()
                        user.is_verified = False
                        user.save()

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
                            return Response({'message': 'OTP sent successfully.'}, template_name='otp_verify.html', status=status.HTTP_403_FORBIDDEN)

                        return Response({'error': 'Failed to send OTP. Please try again.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, template_name='otp_request.html')

                except Exception as e:
                    return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, template_name='otp_request.html')
            
        return Response(status=status.HTTP_200_OK, template_name='otp_request.html')
    

    @action(methods=['get', 'post'], detail=False, url_name='otp_verify', url_path='otp_verify.html', renderer_classes = [TemplateHTMLRenderer])
    def otp_verify(self, request):
        conte = request.session.get('user_data')
        print(conte)
        print(request.user)
        print(request.headers)
        if request.method == 'GET':
            context = {}
            if request.user.is_authenticated: # added check
                context['serial_no'] = request.session.get('serial_no') # added
                context['phone_number'] =request.session.get('phone_number') # added
            else:
                return redirect('users-signin') # added
            return Response(context, status=status.HTTP_200_OK, template_name='otp_verify.html')

        print(request.data)
        if request.method == 'POST':
            serializer = PractitionerSerializer(data=request.data)

            if not serializer.is_valid():
                return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST, template_name='otp_verify.html')
            
            otp = serializer.validated_data['otp']
            serial_no = serializer.validated_data['serial_no']
            phone_number = serializer.validated_data['phone']
            try:
                user = get_object_or_404(CustomUser, serial_no=serial_no, phone=phone_number)
                pract_user = get_object_or_404(Practitioner, user=user, otp=otp)
                if timezone.now() - user.created_at > timezone.timedelta(minutes=15):
                    return Response({'error': 'OTP has expired.'}, status=status.HTTP_400_BAD_REQUEST, template_name='otp_verify.html')

                
                # Verify user and clear OTP
                pract_user.verified = True
                pract_user.otp = None
                pract_user.save()
                return Response({'message': 'OTP verified successfully.', **pract_user}, status=status.HTTP_200_OK, template_name='home.html')

            except CustomUser.DoesNotExist:
                return Response({'error': 'Invalid OTP or phone number.'}, status=status.HTTP_400_BAD_REQUEST, template_name='otp_verify.html')

    @action(methods=['get', 'post'], detail=False, url_name='signup', url_path='signup.html', permission_classes=[AllowAny], renderer_classes = [TemplateHTMLRenderer])
    def signup(self, request):
        from dashboard.models import GENDER_CHOICES
        branches = Branch.objects.all().order_by('name')  # Get all branches
        gender = GENDER_CHOICES[:]
        context = {'branches': branches, 'gender': gender}
        print(request.data)

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
                    # logout(request, user)
                    login(request, user)
                    practitioner_user = Practitioner.objects.create(user=user, otp = str(secrets.randbelow(900000) + 100000), created_at=timezone.now())
                    practitioner_user.save()

                    context['serial_no'] = practitioner_user.user.serial_no
                    context['phone_number'] = practitioner_user.user.phone
                    request.user = user
                    # request.session['otp_data'] = context
                    request.session['user_data'] = user.serial_no
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
                        print(request.session.get('user_data'))
                        return Response({'phone_number': phone_number, 'serial_no': serial_no}, template_name='otp_verify.html', status=status.HTTP_200_OK)
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
            serializer = PatientSerializer(request.data)
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
            serializer = ExaminationSerializer(request.data)

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
            serializer = ClinicalResultSerializer(request.data)

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
            serializer = ClinicalResultSerializer(request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='lab_results.html')
            
            patient = serializer.save()

            return Response({'patient': patient, 'next_step': 'lab-predict', 'next_name': 'Predict Diagnosis', 'message': 'Lab results recorded successfully.'}, status=status.HTTP_201_CREATED, template_name='confirmation.html')      
        if request.method == 'GET':
            patient = request.session.get('patient')   
            # patient = request.data
            print(patient)    
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='lab_results.html')

    @action(methods=['get', 'post'], detail=False, url_name='predict', url_path='predict.html', renderer_classes = [TemplateHTMLRenderer])
    def predict(self, request):
        if request.method == 'POST':
            serializer = ClinicalResultSerializer(request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='predict.html')
            patient = serializer.save()

            serial_no = serializer.validated_data['serial_no']
            user = get_object_or_404(Patient, serial_no=serial_no)
            lab = get_object_or_404(LabResults, patient=serial_no).order_by('-created_at').first()
            clinic = get_object_or_404(ClinicalResult, patient=serial_no).order_by('-created_at').first()
            exam = get_object_or_404(Examination, patient=serial_no).order_by('-created_at').first()
            dob = user.dob.date()  # Convert to `date` object if it's a datetime
            clinic_date = clinic.date.date()  # Same here

            age = clinic_date.year - dob.year - ((clinic_date.month, clinic_date.day) < (dob.month, dob.day))
            gender = user.gender
            bp = exam.bp
            chol = lab.chol_level
            ex_habits = exam.ex_habits
            smoking = exam.smoking_habits
            fam = exam.family_history
            diab = clinic.diabetes
            bmi = clinic.bmi
            hpb = clinic.hpb
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
            # attributes for predictions to be successful predictions
            columns = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
                        'Exercise Habits', 'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI',
                        'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
                        'Alcohol Consumption', 'Stress Level', 'Sleep Hours',
                        'Sugar Consumption', 'Triglyceride Level', 'Fasting Blood Sugar',
                        'CRP Level', 'Homocysteine Level']
            data_df = pd.DataFrame([[age, gender, bp, chol, ex_habits, smoking, fam, diab, bmi, hpb, low_hdl, high_ldl, alc, stress,sleep, sugar, trig, fbs, crp, homo]], columns=columns)

            # load the model and preprocessor
            norm_model = tf.keras.models.load_model('best_model.h5')
            preprocessor = joblib.load('models/heart_preprocessor.pkl')
            
            # preprocess the data and make predictions
            data_df = preprocessor.transform(data_df)
            probability = norm_model.predict(data_df)
            pred_class = int(probability >= 0.5)
            # classes = ['No', 'Yes']
            class_name = 'No' if pred_class == 0 else 'Yes'
            conf_score = round(probability if pred_class == 1 else 1 - probability, 4)

            pred = Prediction.objects.create(patient=user, prediction_type='disease', confidence_score= conf_score, prediction_class = pred_class, prediction_name=class_name, class_probabilities=f'[{probability}, {1 - probability}]', risk_class=float(pred_class), disease_class=pred_class, date=timezone.now(), created_by=request.user)

            return Response({'pred': pred}, template_name='reports.html', status=status.HTTP_201_CREATED)
        if request.method == 'GET':
            patient = request.session.get('patient')
            # patient = request.data
            print(patient) 
            return Response({'patient': patient}, template_name='predict.html', status=status.HTTP_200_OK)
            

class ECGViewSet(ModelViewSet):
    queryset = ECG.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = ECGSerializer

    @action(methods=['get', 'post'], detail=False, url_name='ecg', url_path='ecg.html', renderer_classes = [TemplateHTMLRenderer])
    def ecg(self, request):
        if request.method == 'POST':
            serializer = ClinicalResultSerializer(request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='ecg.html')
            
            patient = serializer.save()
            return Response({'patient': patient, 'next_step': 'ecg-predict', 'next_name': 'Predict Diagnosis', 'message': 'ECG files recorded successfully.'}, status=status.HTTP_201_CREATED, template_name='confirmation.html')      
        if request.method == 'GET':
            patient = request.session.get('patient')
            # patient = request.data
            print(patient)     
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='ecg.html')
        
    @action(methods=['get', 'post'], detail=False, url_name='predict', url_path='predict.html', renderer_classes = [TemplateHTMLRenderer])
    def predict(self, request):
        ecg_model = tf.keras.models.load_model('models/cnn_ecg_deep.keras') #import the ecg model
        if request.method == 'POST':
            serializer = ClinicalResultSerializer(request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='predict.html')
            
            patient = serializer.save()
            scan_done = serializer.validated_data['scan_type']
            if scan_done == 'ecg':
                scan_content = serializer.validated_data['content_hea']
                scan_dat = serializer.validated_data['content_data']
                sig, attr = wfdb.rdsamp(f'data/scan/ecg/{scan_content}')
                prediction_probablities = ecg_model.predict(np.expand_dims(sig, axis=0))
                prediction_class = np.argmax(prediction_probablities)
                ecg_cols = ['HYP', 'NORM', 'CD', 'MI', 'STTC', 'NN']
                index = prediction_probablities.get_index(prediction_class)
                predicted_class = np.argmax(ecg_model.predict(np.expand_dims(sig, axis=0)))
                print(f'The patient has disease belonging the class {ecg_cols[predicted_class]}')
                
                conf_score = prediction_probablities[predicted_class] * 100
                class_name = ecg_cols[predicted_class]
                risk_class = 0 if class_name == 'NORM' else 1
                dis_diag = predicted_class
                pred = Prediction.objects.create(patient=patient, confidence_score=conf_score, predicted_class=predicted_class, predicted_name=class_name, class_probabilities=prediction_probablities, risk_class=risk_class, disease_class=dis_diag)

            return Response({'pred': pred}, status=status.HTTP_201_CREATED, template_name='reports.html')      
        if request.method == 'GET':
            patient = request.session.get('patient')
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
            serializer = MedicalScan(request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='medical_scan.html')
            
            patient = serializer.save()
            return Response({'patient': patient}, status=status.HTTP_201_CREATED, template_name='prediction.html')           
        if request.method == 'GET':
            patient = request.data
            return Response({'patient': patient, 'next_step': 'scan-predict', 'next_name': 'Predict Diagnosis', 'message': 'Chest X-ray recorded successfully.'}, status=status.HTTP_200_OK, template_name='medical_scan.html')
        
    @action(methods=['get', 'post'], detail=False, url_name='predict', url_path='predict.html', renderer_classes = [TemplateHTMLRenderer])
    def predict(self, request):
        chest_model = tf.keras.models.load_model('models/best_model_4.keras') # load chest model
        if request.method == 'POST':
            serializer = ClinicalResultSerializer(request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='predict.html')
            
            patient = serializer.save()

            scan_done = serializer.validated_data['scan_type']
            if scan_done == 'x_ray':
                scan_content = serializer.validated_data['content']
                img_path = f'data/scan/chest/{scan_content}'
                image = cv2.imread(img_path)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = image / 255.0

                prediction_probablities = chest_model.predict(image)
                prediction_class = np.argmax(prediction_probablities)
                final_classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
                 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
                 'Infiltration', 'Mass', 'No Finding', 'Nodule',
                 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
                index = prediction_probablities.get_index(prediction_class)
                class_name = final_classes[index]
                risk_class = 0 if class_name == 'No Finding' else 1
                dis_diag = prediction_class
                predicted_class = np.argmax(chest_model.predict(image))
                conf_score = prediction_probablities[predicted_class] * 100

                pred = Prediction.objects.create(patient=patient, confidence_score=conf_score, predicted_class=predicted_class, predicted_name=class_name, class_probabilities=prediction_probablities, risk_class=risk_class, disease_class=dis_diag)
            return Response({'pred': pred}, status=status.HTTP_201_CREATED, template_name='confirmation.html')      
        if request.method == 'GET':
            patient = request.session.get('patient')
            # patient = request.data
            print(patient)     
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='predict.html')

class BrachViewSet(ModelViewSet):
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
            serializer = PredictionSerializer(request.data)

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

    @action(methods=['get', 'post'], detail=False, url_name='reports', url_path='reports.html', renderer_classes = [TemplateHTMLRenderer])
    def reports(self, request):
        if request.method == 'POST':
            serializer = ClinicalResultSerializer(request.data)

            if not serializer.is_valid():
                return Response({'errors': serializer._errors}, status=status.HTTP_304_NOT_MODIFIED, template_name='reports.html')
            
            patient = serializer.save()

            return Response({'patient': patient}, status=status.HTTP_201_CREATED, template_name='confirmation.html')           
        if request.method == 'GET':
            patient = request.data
            return Response({'patient': patient}, status=status.HTTP_200_OK, template_name='reports.html')


