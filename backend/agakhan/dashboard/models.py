from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from datetime import date
from django.utils import timezone


GENDER_CHOICES = (
    ('F', 'Female'),
    ('M', 'Male'),
    ('I', 'Intersex'),
    ('O', 'Other')
)
SCAN_TYPE = (
    ('ecg', 'ECG Reading Images'),
    ('mri', 'MRI Scan'),
    ('ct_scans', 'CT-Scan'),
    ('x_ray', 'Chest X-Ray')
)
HABITS_CHOICES = (
    ('low', 'Low'),
    ('moderate', 'Moderate'),
    ('high', 'High')
)
TF_CHOICES = (
    ('false', 'Positive'),
    ('true', 'Negative')
)
YN_CHOICES = (
    ('no', 'No'),
    ('yes', 'Yes')
)
PRED_TYPE = (
    ('ecg', 'ECG Reading Images'),
    ('mri', 'MRI Scan'),
    ('ct_scans', 'CT-Scan'),
    ('x_ray', 'Chest X-Ray'),
    ('lab', 'Lab Test Predictions'),
    ('clinic', 'Clinical Assessment Predictions'),
    ('exam', 'Examination Stage Prediction')
)

class CustomUserManager(BaseUserManager):
    def create_user(self, serial_no, password=None, **extra_fields):
        if not serial_no:
            raise ValueError('The Serial No must be set')
        user = self.model(serial_no=serial_no, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, serial_no, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(serial_no, password, **extra_fields)

class CustomUser(AbstractUser):
    username = None # remove username
    objects = CustomUserManager() # use the custom manager
    phone = models.CharField(max_length=20, null=False)
    branch = models.ForeignKey('Branch', on_delete=models.SET_NULL, null=True)
    serial_no = models.CharField(max_length=200, unique=True, primary_key=True)
    gender = models.CharField(max_length=200, choices=GENDER_CHOICES, default='F')
    

    USERNAME_FIELD = 'serial_no'
    REQUIRED_FIELDS = ['first_name', 'last_name', 'phone']

    def __str__(self):
        return f'{self.first_name} - {self.branch} - {self.serial_no}'
    

class Practitioner(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    otp = models.CharField(max_length=6)
    verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)


    def __str__(self):
        return f'{self.user.first_name} - {self.user.branch} - {self.user.serial_no}'

class Branch(models.Model):
    branch_id = models.CharField(max_length=100, unique=True, null=False, default='NRB')
    name = models.CharField(max_length=100)
    address = models.TextField()

    def __str__(self):
        return f'{self.name} - {self.branch_id}'

class Patient(models.Model):
    serial_no = models.CharField(max_length=20, unique=True, primary_key=True, null=False)
    # user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=255, null=False)
    last_name = models.CharField(max_length=255, null=False)
    email = models.EmailField(unique=True)
    location = models.ForeignKey('Branch', on_delete=models.SET_NULL, null=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='F')
    dob = models.DateField()
    phone = models.CharField(max_length=20, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Practitioner, on_delete=models.SET_NULL, null=True)


    def __str__(self):
        return f'{self.first_name} - {self.location} - {self.serial_no}'


class MedicalScan(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    scan_type = models.CharField(max_length=100, choices=SCAN_TYPE, default='x_ray')
    scan_content = models.ImageField(upload_to='data/scans')
    date = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Practitioner, on_delete=models.SET_NULL, null=True)

class ECG(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    scan_type = models.CharField(max_length=100, choices=SCAN_TYPE, default='ecg')
    scan_hea = models.FileField(upload_to='data/scans/ecg')
    scan_dat = models.FileField(upload_to='data/scans/ecg')
    date = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Practitioner, on_delete=models.SET_NULL, null=True)


class LabResults(models.Model):
    patient = models.ForeignKey('Patient', on_delete=models.SET_NULL, null=True)
    low_hdl = models.BooleanField(default=False)
    high_ldl = models.BooleanField(default=False)
    chol_level = models.CharField(max_length=100)
    trig_level = models.CharField(max_length=100)
    crp_level = models.CharField(max_length=100)
    fasting_blood_sugar = models.CharField(max_length=100)
    homo_level = models.CharField(max_length=100)
    date = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Practitioner, on_delete=models.SET_NULL, null=True)

class ClinicalResult(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    diabetes = models.BooleanField(default=False)
    bmi = models.FloatField(default=0.0)
    hbp = models.BooleanField(default=False)
    date = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Practitioner, on_delete=models.SET_NULL, null=True)

class Examination(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    ex_habits = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    smoking_habits = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    family_history = models.BooleanField(default=False)
    alc_habits = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    avrg_sleep = models.FloatField(default=0.0)
    sugar_cons = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    stress_levels = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    bp = models.FloatField(default=0.0)
    date = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Practitioner, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return f'{self.patient.first_name} - {self.patient.branch} - {self.patient.serial_no}'
    
class Prediction(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    prediction_type = models.CharField(max_length=250, choices=PRED_TYPE, default='ecg')
    confidence_score = models.FloatField(default=0.0)
    predicted_class = models.IntegerField()
    predicted_name = models.CharField(max_length=200)
    classes_probablities = models.CharField(max_length=500)
    risk_class = models.IntegerField()
    disease_class = models.IntegerField()
    date = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(Practitioner, on_delete=models.SET_NULL, null=True)
    
    class Meta:
        unique_together = ['patient', 'prediction_type', 'date']
    
    def __str__(self):
        return f'{self.patient} - {self.prediction_type} - {self.date}'

class Report(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    risk_diagnosis = models.CharField(max_length=200)
    disease_diagnosis = models.CharField(max_length=200)
    recommended_check_up = models.TextField(max_length=5000)
    extra_check_up = models.CharField(max_length=200)
    served_by = models.ForeignKey(Practitioner, on_delete=models.SET_NULL, null=True)
    recommended_treatment = models.TextField(max_length=5000)


