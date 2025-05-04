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
    serial_no = models.CharField(max_length=200, unique=True, default='ABC123', primary_key=True)
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
    # serial_no = models.CharField(max_length=20, unique=True)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    email = models.EmailField()
    location = models.ForeignKey('Branch', on_delete=models.SET_NULL, null=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    dob = models.DateField()
    phone = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f'{self.user.first_name} - {self.user.branch} - {self.user.serial_no}'


class MedicalScan(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    scan_type = models.CharField(max_length=100, choices=SCAN_TYPE, default='ecg')
    scan_content = models.ImageField(upload_to='data/scans')
    results = models.TextField()
    date = models.DateTimeField(auto_now_add=True)

class LabResults(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    low_hdl = models.BooleanField(default=False)
    high_ldl = models.BooleanField(default=False)
    chol_level = models.CharField(max_length=100)
    trig_level = models.CharField(max_length=100)
    crp_level = models.CharField(max_length=100)
    fasting_blood_sugar = models.CharField(max_length=100)
    homo_level = models.CharField(max_length=100)
    date = models.DateTimeField(auto_now_add=True)

class ClinicalResult(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    diabetes = models.BooleanField(default=False)
    bmi = models.FloatField(default=0.0)
    hbp = models.BooleanField(default=False)
    date = models.DateTimeField(auto_now_add=True)

class Examination(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    ex_habits = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    smoking_habits = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    family_history = models.BooleanField(default=False)
    alc_habits = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    avrg_sleep = models.FloatField(default=0.0)
    sugar_cons = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    stress_levels = models.CharField(max_length=50, choices=HABITS_CHOICES, default='low')
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.patient.first_name} - {self.patient.branch} - {self.patient.serial_no}'
    
