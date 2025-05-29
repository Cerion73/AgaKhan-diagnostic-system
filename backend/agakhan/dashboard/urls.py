from django.urls import path, include
import rest_framework.urls
from dashboard.views import PractitionerViewset, RegisterPatientViewSet, BranchViewSet, ExaminationViewSet, ClinicalResultViewSet, LabResultsViewSet, MedicalScanViewSet, ReportViewSet, ECGViewSet, PredictViewSet
from rest_framework.routers import DefaultRouter
import rest_framework

router = DefaultRouter()
router.register(r'users', PractitionerViewset, basename='users')
router.register(r'patient', RegisterPatientViewSet, basename='patient')
router.register(r'branch', BranchViewSet, basename='branch')
router.register(r'med-exam', ExaminationViewSet, basename='med-exam')
router.register(r'clinical', ClinicalResultViewSet, basename='clinical')
router.register(r'lab', LabResultsViewSet, basename='lab')
router.register(r'chest', MedicalScanViewSet, basename='chest')
router.register(r'ecg', ECGViewSet, basename='ecg')
router.register(r'predict', PredictViewSet, basename='predict')
router.register(r'report', ReportViewSet, basename='report')

urlpatterns = [
    path('api/', include(rest_framework.urls)),
    path('api-auth/', include(rest_framework.urls)),
    path('', include(router.urls)),
    path('report/', ReportViewSet.as_view({'get': 'reports'}), name='report-reports')
    # path('api/signup/', PractitionerViewset.as_view({'post': 'signup'})),
    # path('auth/signup/', views.SignUpView.as_view(), name='signup'),
    # path('auth/login/', views.LoginView.as_view(), name='login'),
    # path('branches/', views.BranchListView.as_view(), name='branch-list'),
    # path('scans/', views.MedicalScanCreateView.as_view(), name='scan-create'),
    # path('clinical-results/', views.ClinicalResultCreateView.as_view(), name='clinical-create'),
]