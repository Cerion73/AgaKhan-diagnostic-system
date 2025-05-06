# serializers.py
from rest_framework import serializers
from .models import *
from django.contrib.auth.hashers import make_password


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    confirm_password = serializers.CharField(write_only=True)
    branch = serializers.PrimaryKeyRelatedField(queryset=Branch.objects.all())
    gender = serializers.ChoiceField(choices=GENDER_CHOICES)

    class Meta:
        model = CustomUser
        fields = [
            'serial_no',
            'first_name',
            'last_name',
            'phone',
            'password',
            'confirm_password',
            'branch',
            'gender'
        ]
        # extra_kwargs = {
        #     'first_name': {'required': True},
        #     'last_name': {'required': True},
        # }

    def validate(self, data):
        if data['password'] != data['confirm_password']:
            raise serializers.ValidationError("Passwords do not match")
        return data

    def create(self, validated_data):
        validated_data.pop('confirm_password')
        user = CustomUser.objects.create(
            serial_no=validated_data['serial_no'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
            phone=validated_data['phone'],
            branch=validated_data['branch'],
            password=make_password(validated_data['password'])
        )
        return user

class PractitionerSerializer(serializers.ModelSerializer):
    # password = UserSerializer()
    class Meta:
        model = Practitioner
        fields = '__all__'

class BranchSerializer(serializers.ModelSerializer):
    class Meta:
        model = Branch
        fields = '__all__'

class MedicalScanSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicalScan
        fields = '__all__'

class ClinicalResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClinicalResult
        fields = '__all__'

class LabResultsSerializer(serializers.ModelSerializer):
    class Meta:
        model = LabResults
        fields = '__all__'

class ExaminationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Examination
        fields = '__all__'


class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'

class ReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'
        

class ECGSerializer(serializers.ModelSerializer):
    class Meta:
        model = ECG
        fields = '__all__'
