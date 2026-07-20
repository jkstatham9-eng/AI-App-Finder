package com.example.aiappfinder.data

import androidx.room.TypeConverter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Converters {
    @TypeConverter
    fun fromFloatArray(value: FloatArray?): ByteArray? {
        if (value == null) return null
        val byteBuffer = ByteBuffer.allocate(value.size * 4)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        value.forEach { byteBuffer.putFloat(it) }
        return byteBuffer.array()
    }

    @TypeConverter
    fun toFloatArray(value: ByteArray?): FloatArray? {
        if (value == null) return null
        val byteBuffer = ByteBuffer.wrap(value)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        val floatArray = FloatArray(value.size / 4)
        for (i in floatArray.indices) {
            floatArray[i] = byteBuffer.float
        }
        return floatArray
    }
}
