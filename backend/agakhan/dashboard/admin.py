from django.contrib import admin
from dashboard.models import *
from django.contrib.admin import ModelAdmin
# Register your models here.
class PractitionerAdmin(ModelAdmin):
    list_display = ('first_name', 'last_name', 'branch', 'serial_no')
admin.site.register(Patient)
admin.site.register(Practitioner)
admin.site.register(ClinicalResult)
admin.site.register(Branch)
admin.site.register(MedicalScan)
admin.site.register(CustomUser)
admin.site.register(ECG)
admin.site.register(Prediction)
admin.site.register(Report)
admin.site.register(Examination)
admin.site.register(LabResults)
