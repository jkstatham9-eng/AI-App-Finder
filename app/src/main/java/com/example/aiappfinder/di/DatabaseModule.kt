package com.example.aiappfinder.di

import android.content.Context
import androidx.room.Room
import com.example.aiappfinder.data.AppDao
import com.example.aiappfinder.data.AppDatabase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "app_finder_db"
        ).build()
    }

    @Provides
    fun provideAppDao(database: AppDatabase): AppDao {
        return database.appDao()
    }
}
