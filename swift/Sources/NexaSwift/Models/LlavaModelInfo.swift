//
//  LlavaModelInfo.swift
//  llava-ios
//
//  Created by Nexa AI on 11/4/24.
//

import Foundation


struct LlavaModelInfo : Identifiable {
    let id = UUID()
    let modelName: String
    let url: String
    let projectionUrl: String
}

class LlavaModelInfoList : ObservableObject {
    @Published var models: [LlavaModelInfo]
    
    init(models: [LlavaModelInfo]) {
        self.models = models
    }
}
