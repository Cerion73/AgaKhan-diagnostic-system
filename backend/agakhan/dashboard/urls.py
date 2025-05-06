from django.urls import path, include
import rest_framework.urls
from dashboard.views import PractitionerViewset, RegisterPatientViewSet, BrachViewSet, ExaminationViewSet, ClinicalResultViewSet, LabResultsViewSet, MedicalScanViewSet, 
from rest_framework.routers import DefaultRouter
import rest_framework

router = DefaultRouter()
router.register(r'users', PractitionerViewset, basename='users')
router.register(r'patient', RegisterPatientViewSet, basename='patient')
router.register(r'branch', BrachViewSet, basename='branch')
router.register(r'med-exam', ExaminationViewSet, basename='med-exam')
router.register(r'clinical', ClinicalResultViewSet, basename='clinical')
router.register(r'lab', LabResultsViewSet, basename='lab')
router.register(r'imaging', MedicalScanViewSet, basename='imaging')
router.register(r'ecg', MedicalScanViewSet, basename='ecg')
router.register(r'imaging', MedicalScanViewSet, basename='imaging')

urlpatterns = [
    path('api/', include(rest_framework.urls)),
    path('api-auth/', include(rest_framework.urls)),
    path('', include(router.urls)),
    path('api/signup/', PractitionerViewset.as_view({'post': 'signup'})),
    # path('auth/signup/', views.SignUpView.as_view(), name='signup'),
    # path('auth/login/', views.LoginView.as_view(), name='login'),
    # path('branches/', views.BranchListView.as_view(), name='branch-list'),
    # path('scans/', views.MedicalScanCreateView.as_view(), name='scan-create'),
    # path('clinical-results/', views.ClinicalResultCreateView.as_view(), name='clinical-create'),
]