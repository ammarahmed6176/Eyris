$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.21.21-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;" + $env:PATH
$sdkRoot = "C:\Users\ammar\AppData\Local\Android\Sdk"
$sdkManager = "$sdkRoot\cmdline-tools\latest\bin\sdkmanager.bat"

Write-Host "Checking sdkmanager path: $(Test-Path $sdkManager)"

Write-Host "Running sdkmanager --licenses..."
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $sdkManager
$psi.Arguments = "--sdk_root=`"$sdkRoot`" --licenses"
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.EnvironmentVariables["JAVA_HOME"] = $env:JAVA_HOME

$proc = [System.Diagnostics.Process]::Start($psi)
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 400
    if ($proc.HasExited) { break }
    try {
        $proc.StandardInput.WriteLine("y")
    } catch {
        break
    }
}
$proc.WaitForExit(60000)
Write-Host "Licenses output:"
$proc.StandardOutput.ReadToEnd()

Write-Host "Installing platform-tools, platforms;android-34, build-tools;34.0.0..."
$psi2 = New-Object System.Diagnostics.ProcessStartInfo
$psi2.FileName = $sdkManager
$psi2.Arguments = "--sdk_root=`"$sdkRoot`" platform-tools ""platforms;android-34"" ""build-tools;34.0.0"""
$psi2.UseShellExecute = $false
$psi2.RedirectStandardInput = $true
$psi2.RedirectStandardOutput = $true
$psi2.EnvironmentVariables["JAVA_HOME"] = $env:JAVA_HOME

$proc2 = [System.Diagnostics.Process]::Start($psi2)
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    if ($proc2.HasExited) { break }
    try {
        $proc2.StandardInput.WriteLine("y")
    } catch {
        break
    }
}
$proc2.WaitForExit(300000)
Write-Host "SDK install output:"
$proc2.StandardOutput.ReadToEnd()
Write-Host "Done!"
