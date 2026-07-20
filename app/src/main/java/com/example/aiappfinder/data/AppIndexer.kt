package com.example.aiappfinder.data

import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Drawable
import com.example.aiappfinder.ai.OnnxModelManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AppIndexer @Inject constructor(
    @ApplicationContext private val context: Context,
    private val appDao: AppDao,
    private val modelManager: OnnxModelManager
) {
    suspend fun indexInstalledApps() = withContext(Dispatchers.IO) {
        val pm = context.packageManager
        val apps = pm.getInstalledApplications(PackageManager.GET_META_DATA)
        
        for (appInfo in apps) {
            if (appDao.getAppByPackage(appInfo.packageName) != null) continue
            
            val appName = pm.getApplicationLabel(appInfo).toString()
            val icon = pm.getApplicationIcon(appInfo)
            val bitmap = drawableToBitmap(icon)
            
            // Save icon to internal storage
            val iconFile = File(context.filesDir, "${appInfo.packageName}.png")
            FileOutputStream(iconFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            
            val embedding = modelManager.getVisualEmbedding(bitmap)
            
            val primaryHex = com.example.aiappfinder.util.ColorAnalysis.getPrimaryColor(bitmap)
            val entity = AppEntity(
                packageName = appInfo.packageName,
                appName = appName,
                iconPath = iconFile.absolutePath,
                embedding = embedding,
                primaryColor = primaryHex,
                secondaryColors = "",
                shapes = "",
                tags = com.example.aiappfinder.util.ColorAnalysis.getColorName(primaryHex),
                installTime = context.packageManager.getPackageInfo(appInfo.packageName, 0).firstInstallTime,
                updateTime = context.packageManager.getPackageInfo(appInfo.packageName, 0).lastUpdateTime,
                isSystemApp = (appInfo.flags and ApplicationInfo.FLAG_SYSTEM) != 0,
                launchIntent = pm.getLaunchIntentForPackage(appInfo.packageName)?.toUri(0)
            )
            
            appDao.insertApp(entity)
        }
    }

    private fun drawableToBitmap(drawable: Drawable): Bitmap {
        if (drawable is BitmapDrawable) return drawable.bitmap
        val bitmap = Bitmap.createBitmap(
            drawable.intrinsicWidth.coerceAtLeast(1),
            drawable.intrinsicHeight.coerceAtLeast(1),
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, canvas.width, canvas.height)
        drawable.draw(canvas)
        return bitmap
    }
}
