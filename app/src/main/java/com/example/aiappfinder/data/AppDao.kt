package com.example.aiappfinder.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface AppDao {
    @Query("SELECT * FROM apps")
    fun getAllApps(): Flow<List<AppEntity>>

    @Query("SELECT * FROM apps WHERE packageName = :packageName")
    suspend fun getAppByPackage(packageName: String): AppEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertApp(app: AppEntity)

    @Delete
    suspend fun deleteApp(app: AppEntity)

    @Query("DELETE FROM apps WHERE packageName = :packageName")
    suspend fun deleteByPackage(packageName: String)

    @Query("SELECT * FROM apps")
    suspend fun getAllAppsSync(): List<AppEntity>
}
