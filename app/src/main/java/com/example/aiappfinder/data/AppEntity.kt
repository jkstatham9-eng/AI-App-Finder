package com.example.aiappfinder.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "apps")
data class AppEntity(
    @PrimaryKey val packageName: String,
    val appName: String,
    val iconPath: String,
    val embedding: FloatArray,
    val primaryColor: String,
    val secondaryColors: String, // Comma separated
    val shapes: String, // Comma separated
    val tags: String, // Comma separated
    val installTime: Long,
    val updateTime: Long,
    val isSystemApp: Boolean,
    val launchIntent: String?
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as AppEntity
        return packageName == other.packageName
    }

    override fun hashCode(): Int {
        return packageName.hashCode()
    }
}
