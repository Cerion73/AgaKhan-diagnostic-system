# Generated by Django 5.1.6 on 2025-05-05 11:33

import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='Branch',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('branch_id', models.CharField(default='NRB', max_length=100, unique=True)),
                ('name', models.CharField(max_length=100)),
                ('address', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='CustomUser',
            fields=[
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('is_superuser', models.BooleanField(default=False, help_text='Designates that this user has all permissions without explicitly assigning them.', verbose_name='superuser status')),
                ('first_name', models.CharField(blank=True, max_length=150, verbose_name='first name')),
                ('last_name', models.CharField(blank=True, max_length=150, verbose_name='last name')),
                ('email', models.EmailField(blank=True, max_length=254, verbose_name='email address')),
                ('is_staff', models.BooleanField(default=False, help_text='Designates whether the user can log into this admin site.', verbose_name='staff status')),
                ('is_active', models.BooleanField(default=True, help_text='Designates whether this user should be treated as active. Unselect this instead of deleting accounts.', verbose_name='active')),
                ('date_joined', models.DateTimeField(default=django.utils.timezone.now, verbose_name='date joined')),
                ('phone', models.CharField(max_length=20)),
                ('serial_no', models.CharField(max_length=200, primary_key=True, serialize=False, unique=True)),
                ('gender', models.CharField(choices=[('F', 'Female'), ('M', 'Male'), ('I', 'Intersex'), ('O', 'Other')], default='F', max_length=200)),
                ('groups', models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_set', related_query_name='user', to='auth.group', verbose_name='groups')),
                ('user_permissions', models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_set', related_query_name='user', to='auth.permission', verbose_name='user permissions')),
                ('branch', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='dashboard.branch')),
            ],
            options={
                'verbose_name': 'user',
                'verbose_name_plural': 'users',
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Patient',
            fields=[
                ('serial_no', models.CharField(default='AB12', max_length=20, primary_key=True, serialize=False, unique=True)),
                ('first_name', models.CharField(max_length=255)),
                ('last_name', models.CharField(default='', max_length=255)),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('gender', models.CharField(choices=[('F', 'Female'), ('M', 'Male'), ('I', 'Intersex'), ('O', 'Other')], max_length=1)),
                ('dob', models.DateField()),
                ('phone', models.CharField(max_length=20, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('location', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='dashboard.branch')),
            ],
        ),
        migrations.CreateModel(
            name='MedicalScan',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('scan_type', models.CharField(choices=[('ecg', 'ECG Reading Images'), ('mri', 'MRI Scan'), ('ct_scans', 'CT-Scan'), ('x_ray', 'Chest X-Ray')], default='x_ray', max_length=100)),
                ('scan_content', models.ImageField(upload_to='data/scans')),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='dashboard.patient')),
            ],
        ),
        migrations.CreateModel(
            name='LabResults',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('low_hdl', models.BooleanField(default=False)),
                ('high_ldl', models.BooleanField(default=False)),
                ('chol_level', models.CharField(max_length=100)),
                ('trig_level', models.CharField(max_length=100)),
                ('crp_level', models.CharField(max_length=100)),
                ('fasting_blood_sugar', models.CharField(max_length=100)),
                ('homo_level', models.CharField(max_length=100)),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='dashboard.patient')),
            ],
        ),
        migrations.CreateModel(
            name='Examination',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ex_habits', models.CharField(choices=[('low', 'Low'), ('moderate', 'Moderate'), ('high', 'High')], default='low', max_length=50)),
                ('smoking_habits', models.CharField(choices=[('low', 'Low'), ('moderate', 'Moderate'), ('high', 'High')], default='low', max_length=50)),
                ('family_history', models.BooleanField(default=False)),
                ('alc_habits', models.CharField(choices=[('low', 'Low'), ('moderate', 'Moderate'), ('high', 'High')], default='low', max_length=50)),
                ('avrg_sleep', models.FloatField(default=0.0)),
                ('sugar_cons', models.CharField(choices=[('low', 'Low'), ('moderate', 'Moderate'), ('high', 'High')], default='low', max_length=50)),
                ('stress_levels', models.CharField(choices=[('low', 'Low'), ('moderate', 'Moderate'), ('high', 'High')], default='low', max_length=50)),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='dashboard.patient')),
            ],
        ),
        migrations.CreateModel(
            name='ECG',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('scan_type', models.CharField(choices=[('ecg', 'ECG Reading Images'), ('mri', 'MRI Scan'), ('ct_scans', 'CT-Scan'), ('x_ray', 'Chest X-Ray')], default='ecg', max_length=100)),
                ('scan_hea', models.ImageField(upload_to='data/scans/ecg')),
                ('scan_dat', models.ImageField(upload_to='data/scans/ecg')),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='dashboard.patient')),
            ],
        ),
        migrations.CreateModel(
            name='ClinicalResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('diabetes', models.BooleanField(default=False)),
                ('bmi', models.FloatField(default=0.0)),
                ('hbp', models.BooleanField(default=False)),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='dashboard.patient')),
            ],
        ),
        migrations.CreateModel(
            name='Practitioner',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('otp', models.CharField(max_length=6)),
                ('verified', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('confidence_score', models.FloatField(default=0.0)),
                ('predicted_class', models.IntegerField()),
                ('predicted_name', models.CharField(max_length=200)),
                ('classes_probablities', models.CharField(max_length=500)),
                ('risk_class', models.IntegerField()),
                ('disease_class', models.IntegerField()),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='dashboard.patient')),
            ],
        ),
        migrations.CreateModel(
            name='Report',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('risk_diagnosis', models.CharField(max_length=200)),
                ('disease_diagnosis', models.CharField(max_length=200)),
                ('recommended_check_up', models.TextField(max_length=5000)),
                ('extra_check_up', models.CharField(max_length=200)),
                ('recommended_treatment', models.TextField(max_length=5000)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='dashboard.patient')),
                ('served_by', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='dashboard.practitioner')),
            ],
        ),
    ]
