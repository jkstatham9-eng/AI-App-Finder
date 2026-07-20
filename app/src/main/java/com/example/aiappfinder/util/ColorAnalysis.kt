package com.example.aiappfinder.util

import android.graphics.Bitmap
import androidx.palette.graphics.Palette

object ColorAnalysis {
    fun getPrimaryColor(bitmap: Bitmap): String {
        val palette = Palette.from(bitmap).generate()
        val dominant = palette.dominantSwatch
        return dominant?.let {
            String.format("#%06X", (0xFFFFFF and it.rgb))
        } ?: "#FFFFFF"
    }

    fun getColorName(hex: String): String {
        // Simplified color naming logic
        return when {
            hex.startsWith("#FF0000") -> "Red"
            hex.startsWith("#00FF00") -> "Green"
            hex.startsWith("#0000FF") -> "Blue"
            else -> "Other"
        }
    }
}
