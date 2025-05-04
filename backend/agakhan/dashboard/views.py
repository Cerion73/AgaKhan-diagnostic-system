from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.decorators import action
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticatedOrReadOnly, AllowAny
from dashboard.models import Patient, Practitioner, MedicalScan, ClinicalResult, Branch, CustomUser, Examination, LabResults
from dashboard.serializers import PractitionerSerializer, PatientSerializer, BranchSerializer, ClinicalResultSerializer, MedicalScanSerializer, UserSerializer, LabResultsSerializer, ExaminationSerializer
from django.shortcuts import get_object_or_404
from random import randint
from django.utils import timezone
import requests
import uuid
import json
from agakhan.settings import TIARA_CONNECT_API_KEY
import secrets
from django.db import transaction
from django.db.utils import IntegrityError
# from django.contrib.auth import authenticate
# Create your views here.

class PractitionerViewset(ModelViewSet):
    queryset = Practitioner.objects.all()
    # permission_classes = [AllowAny]
    serializer_class = PractitionerSerializer

    @action(methods=['get', 'post'], detail=False, url_name='signin', url_path='signin.html', renderer_classes = [TemplateHTMLRenderer])
    def signin(self, request):
        if request.method == 'POST':
            serializer = PractitionerSerializer(data=request.data)

            if serializer.is_valid():
                password = serializer.validated_data['password']
                serial_no = serializer.validated_data['serial_no']
                
                user = get_object_or_404(Practitioner, serial_no=serial_no, password=password)
                # user = Practitioner.objects.get(serial_no=serial_no, password=password)
                phone_number = user.user.phone

                try:
                    with transaction.atomic():
                        otp = str(secrets.randbelow(900000) + 100000)
                        user.otp = otp
                        user.created_at = timezone.now()
                        user.is_verified = False
                        user.save()

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

                        response = requests.post(
                            'https://api2.tiaraconnect.io/api/messaging/sendsms',
                            json=payload,
                            headers=headers
                        )

                        if response.status_code == 200:
                            return Response({'message': 'OTP sent successfully.'}, template_name='otp_verify.html')

                        return Response({'error': 'Failed to send OTP. Please try again.'},
                                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                        template_name='otp_request.html')

                except Exception as e:
                    # Optional logging here
                    return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, template_name='signup.html')
                return Response({'message': 'Sign-in successful, welcome!'}, template_name='otp_verify.html',status=status.HTTP_200_OK)
        return Response(status=status.HTTP_403_FORBIDDEN, template_name='signin.html')
        
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
        if request.method == 'POST':
            user = request.user()
            serializer = PractitionerSerializer(data=request.data)

            if not serializer.is_valid():
                return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST, template_name='otp_verify.html')
            
            otp = serializer.validated_data['otp']
            try:
                user = Practitioner.objects.get(user=request.user, otp=otp)
                if timezone.now() - user.created_at > timezone.timedelta(minutes=15):
                    return Response({'error': 'OTP has expired.'}, status=status.HTTP_400_BAD_REQUEST, template_name='otp_verify.html')

                
                # Verify user and clear OTP
                user.verified = True
                user.otp = None
                user.created_at = None
                user.save()
                return Response({'message': 'OTP verified successfully.'}, status=status.HTTP_200_OK, template_name='home.html')

            except CustomUser.DoesNotExist:
                return Response({'error': 'Invalid OTP or phone number.'}, status=status.HTTP_400_BAD_REQUEST, template_name='otp_verify.html')
        return Response(status=status.HTTP_200_OK, template_name='otp_verify.html')

    @action(methods=['get', 'post'], detail=False, url_name='signup', url_path='signup.html', permission_classes=[AllowAny], renderer_classes = [TemplateHTMLRenderer])
    def signup(self, request):
        from dashboard.models import GENDER_CHOICES
        branches = Branch.objects.all().order_by('name')  # Get all branches
        gender = GENDER_CHOICES[:]
        context = {'branches': branches, 'gender': gender}
        data = dict(request.data)
        print(data)
        print(request.data)
        print(gender[1:])

        if request.method == 'POST':
            serializer = UserSerializer(data=request.data)
            if not serializer.is_valid():
                print(serializer.error_messages)
                print(serializer._errors)
                return Response({'errors': serializer.errors},template_name='signup.html')

            phone_number = serializer.validated_data['phone']
            try:
                with transaction.atomic():
                    user = serializer.save()
                    user = Practitioner.objects.create(user=user)
                    otp = str(secrets.randbelow(900000) + 100000)
                    user.otp = otp
                    user.created_at = timezone.now()
                    user.is_verified = False
                    user.save()

                    context['serial_no'] = user.user.serial_no
                    context['phone'] = phone_number
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

                    response = requests.post(
                        'https://api2.tiaraconnect.io/api/messaging/sendsms',
                        json=payload,
                        headers=headers
                    )
                    print(response)

                    if response.status_code == 200:
                        return Response({'message': 'OTP sent successfully.'}, context, template_name='otp_verify.html', status=status.HTTP_200_OK)
                    else:
                        return Response({'error': 'Failed to send OTP. Please try again.'}, context,
                                        template_name='otp_request.html', status=status.HTTP_403_FORBIDDEN)

            except Exception as e:
                return Response({'error': str(e)}, template_name='signup.html', status=status.HTTP_403_FORBIDDEN)
            except IntegrityError as e:
                return Response(
                    {'error': 'Serial number or phone already exists.'},
                    status=status.HTTP_400_BAD_REQUEST, 
                    template_name='signup.html'
                )
        return Response(context, template_name='signup.html')

class RegisterPatientViewSet(ModelViewSet):
    queryset = Patient.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = PatientSerializer

class MedicalScanViewSet(ModelViewSet):
    queryset = MedicalScan.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = MedicalScanSerializer

class LabResultsViewSet(ModelViewSet):
    queryset = LabResults.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = LabResultsSerializer

class ClinicalResultViewSet(ModelViewSet):
    queryset = ClinicalResult.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = ClinicalResultSerializer

class ExaminationViewSet(ModelViewSet):
    queryset = Examination.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = ExaminationSerializer

class BrachViewSet(ModelViewSet):
    queryset = Branch.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = BranchSerializer


