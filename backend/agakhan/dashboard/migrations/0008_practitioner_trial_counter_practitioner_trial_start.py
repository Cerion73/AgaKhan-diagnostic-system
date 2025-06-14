# Generated by Django 5.1.6 on 2025-05-17 22:08

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0007_rename_classes_probablities_prediction_classes_probabilities'),
    ]

    operations = [
        migrations.AddField(
            model_name='practitioner',
            name='trial_counter',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='practitioner',
            name='trial_start',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
