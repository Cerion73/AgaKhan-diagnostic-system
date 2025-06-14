# Generated by Django 5.1.6 on 2025-05-07 13:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0005_alter_clinicalresult_diabetes_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='examination',
            name='alc_habits',
            field=models.CharField(choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], default='low', max_length=50),
        ),
        migrations.AlterField(
            model_name='examination',
            name='ex_habits',
            field=models.CharField(choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], default='low', max_length=50),
        ),
        migrations.AlterField(
            model_name='examination',
            name='stress_levels',
            field=models.CharField(choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], default='low', max_length=50),
        ),
        migrations.AlterField(
            model_name='examination',
            name='sugar_cons',
            field=models.CharField(choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], default='low', max_length=50),
        ),
    ]
