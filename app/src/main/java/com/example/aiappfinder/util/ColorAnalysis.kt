package com.example.aiappfinder.util

import android.graphics.Bitmap

object ColorAnalysis {
    fun getPrimaryColor(bitmap: Bitmap): String {
        // Simple color extraction from center pixels
        val width = bitmap.width
        val height = bitmap.height
        val pixel = bitmap.getPixel(width / 2, height / 2)
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        return String.format("#%02X%02X%02X", r, g, b)
    }

    fun getColorName(hex: String): String {
        return when {
            hex.startsWith("#FF0000") -> "Red"
            hex.startsWith("#00FF00") -> "Green"
            hex.startsWith("#0000FF") -> "Blue"
            else -> "Other"
        }
    }
}
