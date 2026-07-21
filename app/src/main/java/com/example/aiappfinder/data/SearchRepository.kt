package com.example.aiappfinder.data

import com.example.aiappfinder.ai.OnnxModelManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.sqrt

@Singleton
class SearchRepository @Inject constructor(
    private val appDao: AppDao,
    private val modelManager: OnnxModelManager
) {
    fun getAllApps(): Flow<List<AppEntity>> = appDao.getAllApps()

    suspend fun searchApps(query: String): List<AppEntity> = withContext(Dispatchers.Default) {
        val queryEmbedding = modelManager.getTextEmbedding(query)
        val allApps = appDao.getAllAppsSync()

        val results = allApps.map { app ->
            val similarity = cosineSimilarity(queryEmbedding, app.embedding)
            app to similarity
        }.sortedByDescending { it.second }
            .map { it.first }

        // Dispose text model session after search is complete
        modelManager.disposeAllSessions()

        results
    }

    suspend fun findSimilarApps(targetApp: AppEntity): List<AppEntity> = withContext(Dispatchers.Default) {
        val allApps = appDao.getAllAppsSync()
        val results = allApps.filter { it.packageName != targetApp.packageName }
            .map { app ->
                val similarity = cosineSimilarity(targetApp.embedding, app.embedding)
                app to similarity
            }.sortedByDescending { it.second }
            .map { it.first }

        results
    }

    private fun cosineSimilarity(vec1: FloatArray, vec2: FloatArray): Float {
        var dotProduct = 0.0f
        var normA = 0.0f
        var normB = 0.0f
        for (i in vec1.indices) {
            dotProduct += vec1[i] * vec2[i]
            normA += vec1[i] * vec1[i]
            normB += vec2[i] * vec2[i]
        }
        return dotProduct / (sqrt(normA) * sqrt(normB))
    }
}
