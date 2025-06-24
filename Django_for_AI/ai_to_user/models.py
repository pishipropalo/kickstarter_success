from django.db import models

# Create your models here.

class Project(models.Model):
    name = models.CharField(max_length=200, verbose_name="Название проекта")
    blurb = models.TextField(verbose_name="Краткое описание")
    country = models.CharField(max_length=50, verbose_name="Страна")
    usd_goal = models.FloatField(verbose_name="Финансовая цель (USD)")
    campaign_days = models.IntegerField(verbose_name="Длина кампании (дни)")
    prelaunch_activated = models.BooleanField(verbose_name="Предстартовая кампания активирована?")
    creation_date = models.DateField(verbose_name="Дата создания")
    creation_time = models.TimeField(verbose_name="Время создания")
    launch_date = models.DateField(verbose_name="Дата запуска")
    launch_time = models.TimeField(verbose_name="Время запуска")
    description = models.TextField(verbose_name="Текстовое описание")
    # Здесь мы используем TextField для ссылок на изображения, т.к. их может быть несколько.
    # Если бы была только одна основная картинка, использовали бы ImageField.
    images = models.TextField(
        blank=True, verbose_name="Ссылки на изображения проекта (через запятую)"
    )
    video = models.URLField(blank=True, null=True, verbose_name="Ссылка на видео ")

    class Meta:
        verbose_name = "Проект"
        verbose_name_plural = "Проекты"

    def __str__(self):
        # Эта строка должна ВСЕГДА возвращать строку.
        # Если name не существует, это может вызвать ошибку, но обычно name есть.
        return self.name