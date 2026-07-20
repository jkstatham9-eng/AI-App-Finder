package com.example.aiappfinder.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.aiappfinder.data.AppEntity
import com.example.aiappfinder.data.AppIndexer
import com.example.aiappfinder.data.SearchRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class MainViewModel @Inject constructor(
    private val repository: SearchRepository,
    private val indexer: AppIndexer
) : ViewModel() {

    private val _searchQuery = MutableStateFlow("")
    val searchQuery = _searchQuery.asStateFlow()

    private val _isIndexing = MutableStateFlow(false)
    val isIndexing = _isIndexing.asStateFlow()

    val apps: StateFlow<List<AppEntity>> = searchQuery
        .debounce(300)
        .flatMapLatest { query ->
            if (query.isBlank()) {
                repository.getAllApps()
            } else {
                flow { emit(repository.searchApps(query)) }
            }
        }
        .stateIn(viewModelScope, SharingStarted.Lazily, emptyList())

    init {
        startIndexing()
    }

    fun onSearchQueryChange(query: String) {
        _searchQuery.value = query
    }

    private fun startIndexing() {
        viewModelScope.launch {
            _isIndexing.value = true
            indexer.indexInstalledApps()
            _isIndexing.value = false
        }
    }
}
